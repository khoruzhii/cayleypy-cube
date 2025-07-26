import torch
import json
import math
import numpy as np

def load_cube_data(cube_size, cube_type, device):
    """Load cube data based on cube size and type (qtm or all)."""
    file_path = f"generators/{cube_type}_cube{cube_size}.json"
    
    with open(file_path, 'rb') as f:
        data = json.load(f)
    
    actions = data["actions"]
    action_names = data["names"]
    
    return torch.tensor(actions, dtype=torch.int64, device=device), action_names

def generate_inverse_moves(moves):
    """Generate the inverse moves for a given list of moves."""
    inverse_moves = [0] * len(moves)
    for i, move in enumerate(moves):
        if "'" in move:  # It's an a_j'
            inverse_moves[i] = moves.index(move.replace("'", ""))
        else:  # It's an a_j
            inverse_moves[i] = moves.index(move + "'")
    return inverse_moves

def state2hash(states, hash_vec, batch_size=2**14):
    """Convert states to hashes."""
    num_batches = (states.size(0) + batch_size - 1) // batch_size
    result = torch.empty(states.size(0), dtype=torch.int64, device=states.device)
    
    for i in range(num_batches):
        batch = states[i * batch_size:(i + 1) * batch_size].to(torch.int64)
        batch_hash = torch.sum(hash_vec * batch, dim=1)
        result[i * batch_size:(i + 1) * batch_size] = batch_hash
    return result

# sliding puzzle specific
def get_mask_periodic(states):
    n = int(math.sqrt(2 + states.size(1))-1)
    mask_periodic = torch.ones((states.size(0), 4), dtype=torch.bool, device=states.device)
    mask_periodic[states[:, -1  ] == 0, 2] = False
    mask_periodic[states[:, -n  ] == 0, 3] = False
    mask_periodic[states[:, -n-1] == 0, 0] = False
    mask_periodic[states[:, -n-n] == 0, 1] = False
    return mask_periodic

# sliding puzzle specific
def state2im(state):
    n = int(math.sqrt(2 + state.size(0))-1)
    M = np.zeros((2*n-1,2*n-1), dtype=int)
    M[n-1:, n-1:] = np.concatenate(([0], state[:n**2-1])).reshape(n,n)
    M[np.arange(n-1)] = M[np.arange(n-1)+n]
    M[:, np.arange(n-1)] = M[:, np.arange(n-1)+n]
    idx_hor, idx_ver = n-1-state[n*n-1], n-1-state[n*(n+1)-1]
    return M[idx_ver:idx_ver+n, idx_hor:idx_hor+n]