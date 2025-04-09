import torch
import time
from collections import deque
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
        self.hash_vec = torch.randint(0, int(1e15), (self.state_size,), device=self.device, dtype=torch.int64)
        self.verbose = verbose
    
    def get_unique_states(self, states, states_bad_hashed):
        """Filter unique states by removing duplicates based on hash."""
        idx1 = torch.arange(states.size(0), dtype=torch.int64, device=states.device)
        hashed = state2hash(states, self.hash_vec, self.batch_size)
        mask1  = ~torch.isin(hashed, states_bad_hashed)
        hashed = hashed[mask1]
        hashed_sorted, idx2 = torch.sort(hashed)
        mask2 = torch.concat((torch.tensor([True], device=states.device), hashed_sorted[1:] - hashed_sorted[:-1] > 0))
        return states[mask1][idx2[mask2]], idx1[mask1][idx2[mask2]] 
    
    def get_unique_hashed_states_idx(self, hashed, states_bad_hashed):
        """Filter unique hashed states by removing duplicates"""
        idx1 = torch.arange(hashed.size(0), dtype=torch.int64, device=hashed.device)
        mask1  = ~torch.isin(hashed, states_bad_hashed)
        hashed = hashed[mask1]
        hashed_sorted, idx2 = torch.sort(hashed)
        mask2 = torch.concat((torch.tensor([True], device=hashed.device), hashed_sorted[1:] - hashed_sorted[:-1] > 0))
        return idx1[mask1][idx2[mask2]] 
    
    def get_neighbors(self, states):
        """Return neighboring states for each state in the batch."""
        neighbors = torch.empty(states.size(0), self.n_gens, self.state_size, device=self.device, dtype=states.dtype)
        for i in range(0, states.size(0), self.batch_size):
            batch_states = states[i:i + self.batch_size]
            neighbors[i:i + self.batch_size] = torch.gather(
                batch_states.unsqueeze(1).expand(batch_states.size(0), self.n_gens, self.state_size), 
                2, 
                self.all_moves.unsqueeze(0).expand(batch_states.size(0), self.n_gens, self.state_size)
            )
        return neighbors
    
    def apply_move(self, states, moves):
        moved_states = torch.empty(states.size(0), self.state_size, device=self.device, dtype=states.dtype)
        for i in range(0, states.size(0), self.batch_size):
            moved_states[i:i+self.batch_size] = torch.gather(states[i:i+self.batch_size], 1, self.all_moves[moves[i:i+self.batch_size]])
        return moved_states
    
    def do_greedy_step(self, states, states_bad_hashed, B=1000):
        """Perform a greedy step to find the best neighbors."""
        idx0 = torch.arange(states.size(0), device=self.device).repeat_interleave(self.n_gens)
        moves = torch.arange(self.n_gens, device=self.device).repeat(states.size(0))

        neighbors_hashed = torch.empty(moves.size(0), dtype=torch.int64, device=self.device)
        for i in range(0, states.size(0), self.batch_size):
            batch_states = states[i:i+self.batch_size]
            neighbors = self.get_neighbors(batch_states).flatten(end_dim=1)
            neighbors_hashed[i*self.n_gens:(i+self.batch_size)*self.n_gens] = state2hash(neighbors, self.hash_vec, self.batch_size)
        idx1 = self.get_unique_hashed_states_idx(neighbors_hashed, states_bad_hashed)
        
        value = torch.empty(idx1.size(0), dtype=torch.float16, device=self.device)
        for i in range(0, idx1.size(0), self.batch_size):
            batch_states = self.apply_move(states[idx0[idx1[i:i+self.batch_size]]], moves[idx1[i:i+self.batch_size]])
            value[i:i+self.batch_size] = self.pred_d(batch_states)[0]
        idx2 = torch.argsort(value)[:B]
        
        next_states = torch.empty(idx2.size(0), self.state_size, dtype=states.dtype, device=self.device)
        for i in range(0, idx2.size(0), self.batch_size):
            next_states[i:i+self.batch_size] = self.apply_move(
                states[idx0[idx1[idx2[i:i+self.batch_size]]]], 
                moves[idx1[idx2[i:i+self.batch_size]]])

        return next_states, value[idx2], moves[idx1[idx2]], idx0[idx1[idx2]]
    
    def check_stagnation(self, states_log):
        """Check if the process is in a stagnation state."""
        return torch.isin(torch.concat(list(states_log)[2:]), torch.concat(list(states_log)[:2])).all().item()

    
    def get_solution(self, state, B=2**12, num_steps=200, num_attempts=10, return_tree=False):
        """Main solution-finding loop that attempts to solve the cube."""
        states_bad_hashed = torch.tensor([], dtype=torch.int64, device=self.device)
        for J in range(num_attempts):
            states = state.unsqueeze(0).clone()
            tree_move = -torch.ones((num_steps, B), dtype=torch.int64)
            tree_idx = -torch.ones((num_steps, B), dtype=torch.int64)
#             tree_value = -torch.ones((num_steps, B), dtype=torch.int64)
            states_hash_log = deque(maxlen=4)
            
            if self.verbose:
                pbar = tqdm(range(num_steps))
            else:
                pbar = range(num_steps)
            for j in pbar:
                states, y_pred, moves, idx = self.do_greedy_step(states, states_bad_hashed, B)
                if self.verbose:
                    pbar.set_description(
                        f"  y_min = {y_pred.min().item():.1f}, y_mean = {y_pred.mean().item():.1f}, y_max = {y_pred.max().item():.1f}"
                    )
                states_hash_log.append(state2hash(states, self.hash_vec))
                leaves_num = states.size(0)
                tree_move[j, :leaves_num] = moves
                tree_idx[j, :leaves_num] = idx

                if (states == self.V0).all(dim=1).any():
                    break
                elif (j > 3 and self.check_stagnation(states_hash_log)):
                    states_bad_hashed = torch.concat((states_bad_hashed, torch.concat(list(states_hash_log))))
                    states_bad_hashed = torch.unique(states_bad_hashed)
                    break

            if (states == self.V0).all(dim=1).any():
                break
        
        if not (states == self.V0).all(dim=1).any():
            return None, J
        
        # Reverse the tree to reconstruct the path
        tree_idx, tree_move = tree_idx[:j+1].flip((0,)), tree_move[:j+1].flip((0,))
        
        V0_pos = torch.nonzero((states == self.V0).all(dim=1), as_tuple=True)[0].item()
        
        # Construct the path
        path = [tree_idx[0, V0_pos].item()]
        for k in range(1, j+1):
            path.append(tree_idx[k, path[-1]].item())
        
        moves_seq = torch.tensor([tree_move[k, path[k-1]] if k > 0 else tree_move[k, V0_pos] for k in range(j+1)], dtype=torch.int64)
        if return_tree:
            return moves_seq.flip((0,)), J, torch.concat((tree_idx.unsqueeze(0), tree_move.unsqueeze(0))).cpu()
        else:
            return moves_seq.flip((0,)), J
    
    def pred_d(self, states):
        """Predict values for states using the model."""
        pred = batch_process(self.model, states, self.device, 2**14)
#         pred[(states == self.V0).all(dim=-1)] = 0
        return pred.unsqueeze(0)
