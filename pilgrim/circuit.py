import torch
import itertools

def get_perm_x(m, device='cpu'):
    # Generate X gate permutations: each row flips one qubit.
    idx = torch.arange(2**m, dtype=torch.long, device=device)
    return (idx[:, None] ^ (1 << torch.arange(m, dtype=torch.long, device=device))).T

def get_perm_cnot(m, device='cpu'):
    # Generate CNOT gate permutations: for each ordered (control, target) pair with control != target.
    idx = torch.arange(2**m, dtype=torch.long, device=device)
    perms = [idx ^ ((1 << target) * ((idx >> ctrl) & 1))
             for ctrl in range(m) for target in range(m) if ctrl != target]
    return torch.stack(perms)

def get_perm_ccnot(m, device='cpu'):
    # Generate CCNOT gate permutations: for each target, choose two control bits from the remaining ones.
    idx = torch.arange(2**m, dtype=torch.long, device=device)
    perms = [idx ^ ((1 << target) * (((idx >> ctrl1) & 1) * ((idx >> ctrl2) & 1)))
             for target in range(m)
             for ctrl1, ctrl2 in itertools.combinations(range(m), 2) if target not in (ctrl1, ctrl2)]
    return torch.stack(perms)

def get_all_moves(m, device='cpu'):
    return torch.cat((get_perm_x(m), get_perm_cnot(m), get_perm_ccnot(m))).to(device)

def get_ic_cnot(m, device='cpu'):
    # Generate initial by CNOT
    idx = torch.arange(2**m, dtype=torch.long, device=device)
    return (idx[:, None] & (1 << torch.arange(m, dtype=torch.long, device=device))).T.clip(0, 1)