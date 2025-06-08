import torch
import time
import os
from collections import deque
from tqdm import tqdm
from .model import batch_process


def state2hash(states, hash_vec, batch_size=2**14):
    """Convert states to 64-bit hashes."""
    n = states.size(0)
    result = torch.empty(n, dtype=torch.int64, device=states.device)
    
    for i in range(0, n, batch_size):
        batch = states[i:i+batch_size].to(torch.int64)
        result[i:i+batch_size] = torch.sum(hash_vec * batch, dim=1)
    
    return result


class Searcher:
    def __init__(self, model, all_moves, V0, device=None, verbose=0, device_eval='cpu'):
        self.model = model.to(device_eval)
        self.all_moves = all_moves
        self.V0 = V0
        self.batch_size = 2**14
        self.n_gens = all_moves.size(0)
        self.state_size = all_moves.size(1)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_eval = device_eval
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.hash_vec = torch.randint(0, int(1e15), (self.state_size,), device=self.device, dtype=torch.int64)
        self.verbose = verbose
        self.counter = torch.zeros((3, 2), dtype=torch.int64)
    
    def state2hash(self, states):
        """Convert states to hashes."""
        return state2hash(states, self.hash_vec, self.batch_size)
    
    def get_unique_hashed_states_idx(self, hashed):
        """Filter unique hashed states by removing duplicates."""
        idx1 = torch.arange(hashed.size(0), dtype=torch.int64, device=hashed.device)
        hashed_sorted, idx2 = torch.sort(hashed)
        mask = torch.cat([
            torch.tensor([True], device=hashed.device),
            hashed_sorted[1:] - hashed_sorted[:-1] > 0
        ])
        return idx1[idx2[mask]]
    
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
    
    def do_greedy_step(self, states, B=1000):
        """Perform a greedy step to find the best neighbors."""
        idx0 = torch.arange(states.size(0), device=self.device).repeat_interleave(self.n_gens)
        moves = torch.arange(self.n_gens, device=self.device).repeat(states.size(0))
        self.counter[0, 0] += moves.size(0); self.counter[0, 1] += 1;

        # Compute hashes for all neighbors
        neighbors_hashed = torch.empty(moves.size(0), dtype=torch.int64, device=self.device)
        for i in range(0, states.size(0), self.batch_size):
            batch_states = states[i:i+self.batch_size]
            neighbors = self.get_neighbors(batch_states).flatten(end_dim=1)
            neighbors_hashed[i*self.n_gens:(i+self.batch_size)*self.n_gens] = self.state2hash(neighbors)
        
        idx1 = self.get_unique_hashed_states_idx(neighbors_hashed)
        self.counter[1, 0] += idx1.size(0); self.counter[1, 1] += 1;
        
        # Evaluate unique states
        value = torch.empty(idx1.size(0), dtype=torch.float16, device=self.device)
        for i in range(0, idx1.size(0), self.batch_size):
            batch_states = self.apply_move(states[idx0[idx1[i:i+self.batch_size]]], moves[idx1[i:i+self.batch_size]])
            value[i:i+self.batch_size] = self.pred_d(batch_states)[0]
        
        # Select best B states
        idx2 = torch.argsort(value)[:B]
        self.counter[2, 0] += idx2.size(0); self.counter[2, 1] += 1;
        
        # Generate next states
        next_states = torch.empty(idx2.size(0), self.state_size, dtype=states.dtype, device=self.device)
        for i in range(0, idx2.size(0), self.batch_size):
            next_states[i:i+self.batch_size] = self.apply_move(
                states[idx0[idx1[idx2[i:i+self.batch_size]]]], 
                moves[idx1[idx2[i:i+self.batch_size]]])

        return next_states, value[idx2], moves[idx1[idx2]], idx0[idx1[idx2]]
    
    def get_solution(self, state, B=2**12, num_steps=200, num_attempts=1, return_tree=False):
        """Main solution-finding loop that attempts to solve the cube."""
        # Create tree directory for temporary storage
        tree_dir = "tree"
        os.makedirs(tree_dir, exist_ok=True)
        
        states = state.unsqueeze(0).clone()
        
        if self.verbose:
            pbar = tqdm(range(num_steps))
        else:
            pbar = range(num_steps)
            
        solved = False
        last_j = -1
        for j in pbar:
            states, y_pred, moves, idx = self.do_greedy_step(states, B)
            if self.verbose:
                pbar.set_description(
                    f"  y_min = {y_pred.min().item():.1f}, y_mean = {y_pred.mean().item():.1f}, y_max = {y_pred.max().item():.1f}"
                )
            
            # Save layer to disk
            layer_data = {
                'moves': moves.cpu(),
                'idx': idx.cpu()
            }
            torch.save(layer_data, os.path.join(tree_dir, f"{j:04d}.pt"))
            last_j = j

            if (states == self.V0).all(dim=1).any():
                solved = True
                break
        
        if not solved:
            # Clean up tree files
            for j in range(last_j + 1):
                filepath = os.path.join(tree_dir, f"{j:04d}.pt")
                if os.path.exists(filepath):
                    os.remove(filepath)
            return None, 0
        
        # Find position of solved state
        V0_pos = torch.nonzero((states == self.V0).all(dim=1), as_tuple=True)[0].item()
        
        # Reconstruct path backwards from solution
        path = [V0_pos]
        moves_list = []
        
        # Load last layer to get the move that led to solution
        layer = torch.load(os.path.join(tree_dir, f"{last_j:04d}.pt"))
        moves_list.append(layer['moves'][V0_pos].item())
        
        # Trace back through tree
        current_pos = V0_pos
        for k in range(last_j - 1, -1, -1):
            # Get parent index from current layer
            current_layer = torch.load(os.path.join(tree_dir, f"{k+1:04d}.pt"))
            parent_pos = current_layer['idx'][current_pos].item()
            
            # Get move from parent layer
            parent_layer = torch.load(os.path.join(tree_dir, f"{k:04d}.pt"))
            moves_list.append(parent_layer['moves'][parent_pos].item())
            
            current_pos = parent_pos
            path.append(current_pos)
        
        # Reverse to get forward path
        moves_list.reverse()
        moves_seq = torch.tensor(moves_list, dtype=torch.int64)
        
        # Clean up tree files
        for j in range(last_j + 1):
            filepath = os.path.join(tree_dir, f"{j:04d}.pt")
            if os.path.exists(filepath):
                os.remove(filepath)
        
        if return_tree:
            return moves_seq, 0, None
        else:
            return moves_seq, 0
    
    def pred_d(self, states):
        """Predict values for states using the model."""
        pred = batch_process(self.model, states, self.device, self.device_eval, 2**14)
        return pred.unsqueeze(0)