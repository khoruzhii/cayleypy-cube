# searcher.py

import torch
import time
from tqdm import tqdm
from .utils import state2hash
from .model import batch_process


class Searcher:
    def __init__(self, model, all_moves, V0, device=None, verbose=0):
        self.model = model.to(device)
        self.all_moves = all_moves
        self.V0 = V0
        self.batch_size = 2**14
        self.n_gens = all_moves.size(0)
        self.state_size = all_moves.size(1)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hash_vec = torch.randint(
            0, int(1e15), (self.state_size,), device=self.device, dtype=torch.int64
        )
        self.verbose = verbose
        # counter[0] – neighbors before dedup, counter[1] – after local dedup, counter[2] – beam size
        # [:, 0] – total count, [:, 1] – number of steps
        self.counter = torch.zeros((3, 2), dtype=torch.int64)

    def get_unique_hashed_states_idx(self, hashed):
        """Return indices of unique hashes (local deduplication)."""
        if hashed.size(0) == 0:
            return torch.empty(0, dtype=torch.int64, device=hashed.device)
        idx_all = torch.arange(hashed.size(0), dtype=torch.int64, device=hashed.device)
        hashed_sorted, idx_sorted = torch.sort(hashed)
        mask_unique = torch.concat(
            (torch.tensor([True], device=hashed.device),
             hashed_sorted[1:] - hashed_sorted[:-1] > 0)
        )
        return idx_all[idx_sorted[mask_unique]]

    def get_neighbors(self, states):
        """Return neighboring states for each state in the batch."""
        neighbors = torch.empty(
            states.size(0), self.n_gens, self.state_size,
            device=self.device, dtype=states.dtype
        )
        for i in range(0, states.size(0), self.batch_size):
            batch_states = states[i:i + self.batch_size]
            neighbors[i:i + batch_states.size(0)] = torch.gather(
                batch_states.unsqueeze(1).expand(batch_states.size(0), self.n_gens, self.state_size),
                2,
                self.all_moves.unsqueeze(0).expand(batch_states.size(0), self.n_gens, self.state_size)
            )
        return neighbors

    def apply_move(self, states, moves):
        """Apply moves to states and return new states."""
        moved_states = torch.empty(
            states.size(0), self.state_size,
            device=self.device, dtype=states.dtype
        )
        for i in range(0, states.size(0), self.batch_size):
            batch_states = states[i:i + self.batch_size]
            batch_moves = moves[i:i + self.batch_size]
            moved_states[i:i + batch_states.size(0)] = torch.gather(
                batch_states, 1, self.all_moves[batch_moves]
            )
        return moved_states

    def do_greedy_step(self, states, B=1000):
        """Perform a greedy step with batched local deduplication and streaming global top-B."""
        device = states.device
        S = states.size(0)

        total_neighbors = S * self.n_gens
        self.counter[0, 0] += total_neighbors
        self.counter[0, 1] += 1

        # First pass: local deduplication in state batches, only collect neighbor ids
        neighbor_ids_chunks = []
        total_after_dedup = 0

        for start_state in range(0, S, self.batch_size):
            end_state = min(start_state + self.batch_size, S)
            bs = end_state - start_state
            if bs <= 0:
                continue

            batch_states = states[start_state:end_state]  # [bs, state_size]
            # neighbors_batch: [bs, n_gens, state_size]
            neighbors_batch = self.get_neighbors(batch_states)
            neighbors_flat = neighbors_batch.flatten(end_dim=1)  # [bs * n_gens, state_size]

            hashed = state2hash(neighbors_flat, self.hash_vec, self.batch_size)
            unique_local = self.get_unique_hashed_states_idx(hashed)

            if unique_local.numel() == 0:
                continue

            total_after_dedup += unique_local.size(0)
            global_offset = start_state * self.n_gens
            neighbor_ids_chunks.append(global_offset + unique_local)

        self.counter[1, 0] += total_after_dedup
        self.counter[1, 1] += 1

        if len(neighbor_ids_chunks) == 0:
            # Fallback: no new neighbors, keep current beam
            dummy_values = torch.full(
                (S,), float("inf"), dtype=torch.float16, device=device
            )
            dummy_moves = torch.zeros(S, dtype=torch.int64, device=device)
            idx0_dummy = torch.arange(S, device=device, dtype=torch.int64)
            return states, dummy_values, dummy_moves, idx0_dummy

        neighbor_ids_all = torch.cat(neighbor_ids_chunks, dim=0)  # [N_unique]

        # Second pass: streaming global top-B over unique neighbors
        best_values = None
        best_neighbor_ids = None

        for start in range(0, neighbor_ids_all.size(0), self.batch_size):
            end = min(start + self.batch_size, neighbor_ids_all.size(0))
            ids_batch = neighbor_ids_all[start:end]

            parent_idx_batch = ids_batch // self.n_gens
            moves_batch = ids_batch % self.n_gens

            parent_states_batch = states[parent_idx_batch]
            neighbor_states_batch = self.apply_move(parent_states_batch, moves_batch)
            values_batch = self.pred_d(neighbor_states_batch)

            if best_values is None:
                best_values = values_batch
                best_neighbor_ids = ids_batch
            else:
                combined_values = torch.cat([best_values, values_batch], dim=0)
                combined_ids = torch.cat([best_neighbor_ids, ids_batch], dim=0)

                if combined_values.size(0) <= B:
                    best_values = combined_values
                    best_neighbor_ids = combined_ids
                else:
                    topk = torch.topk(combined_values, k=B, largest=False)
                    best_values = combined_values[topk.indices]
                    best_neighbor_ids = combined_ids[topk.indices]

        # Now best_neighbor_ids and best_values hold up to B best neighbors
        parent_idx = best_neighbor_ids // self.n_gens
        moves_idx = best_neighbor_ids % self.n_gens

        next_states = self.apply_move(states[parent_idx], moves_idx)

        # Final deduplication over the resulting beam
        hashed_B = state2hash(next_states, self.hash_vec, self.batch_size)
        unique_B = self.get_unique_hashed_states_idx(hashed_B)

        next_states = next_states[unique_B]
        next_values = best_values[unique_B]
        next_moves = moves_idx[unique_B]
        next_idx0 = parent_idx[unique_B]

        self.counter[2, 0] += next_states.size(0)
        self.counter[2, 1] += 1

        return next_states, next_values, next_moves, next_idx0

    def get_solution(self, state, B=2**12, num_steps=200, num_attempts=10, return_tree=False):
        """Main solution-finding loop. num_attempts is kept for compatibility but not used."""
        states = state.unsqueeze(0).clone()
        tree_move = -torch.ones((num_steps, B), dtype=torch.int64)
        tree_idx = -torch.ones((num_steps, B), dtype=torch.int64)
        J = 0  # single attempt

        if self.verbose:
            pbar = tqdm(range(num_steps))
        else:
            pbar = range(num_steps)

        found = False
        for j in pbar:
            states, y_pred, moves, idx = self.do_greedy_step(states, B)
            if self.verbose:
                pbar.set_description(
                    f"  y_min = {y_pred.min().item():.1f}, "
                    f"y_mean = {y_pred.mean().item():.1f}, "
                    f"y_max = {y_pred.max().item():.1f}"
                )

            leaves_num = states.size(0)
            tree_move[j, :leaves_num] = moves
            tree_idx[j, :leaves_num] = idx

            if (states == self.V0).all(dim=1).any():
                found = True
                break

        if not found:
            return None, J

        # Reverse time axis to reconstruct the path
        tree_idx, tree_move = tree_idx[:j + 1].flip((0,)), tree_move[:j + 1].flip((0,))

        V0_pos = torch.nonzero((states == self.V0).all(dim=1), as_tuple=True)[0].item()

        # Reconstruct index path backwards
        path = [tree_idx[0, V0_pos].item()]
        for k in range(1, j + 1):
            path.append(tree_idx[k, path[-1]].item())

        moves_seq = torch.tensor(
            [tree_move[k, path[k - 1]] if k > 0 else tree_move[k, V0_pos]
             for k in range(j + 1)],
            dtype=torch.int64
        )

        if return_tree:
            return moves_seq.flip((0,)), J, torch.concat(
                (tree_idx.unsqueeze(0), tree_move.unsqueeze(0))
            ).cpu()
        else:
            return moves_seq.flip((0,)), J

    def pred_d(self, states):
        """Predict values for states using the model."""
        pred = batch_process(self.model, states, self.device, 2**14)
        return pred
