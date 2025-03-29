import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, layers):
        """
        Constructor:
        - layers: list of layer sizes. The first element is the input dimension and the last is the output dimension.
        """
        super(MLP, self).__init__()
        self.activation = nn.ReLU()
        modules = []
        # Build hidden layers: Linear -> BatchNorm -> ReLU
        for i in range(len(layers) - 2):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(nn.BatchNorm1d(layers[i+1]))
            modules.append(nn.ReLU())
        # Final output layer without BatchNorm or activation
        modules.append(nn.Linear(layers[-2], layers[-1]))
        self.model = nn.Sequential(*modules)
    
    def forward(self, x):
        # x is assumed to be binary input with shape (batch_size, layers[0])
        out = self.model(x)
        return out.flatten() if out.dim() > 1 and out.size(1) == 1 else out

def count_parameters(model):
    """Count the trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def batch_process(model, data, device, batch_size):
    """
    Process data through a model in batches.

    :param data: Tensor of input data
    :param model: A PyTorch model with a forward method that accepts data
    :param device: Device to perform computations (e.g., 'cuda', 'cpu')
    :param batch_size: Number of samples per batch
    :return: Concatenated tensor of model outputs
    """
    model.eval()
    model.to(device)

    outputs = torch.empty(data.size(0), dtype=torch.float32, device=device)

    # Process each batch
    for i in range(0, data.size(0), batch_size):
        batch = data[i:i+batch_size].to(device)
        with torch.no_grad():
            batch_output = model(batch).flatten()
        outputs[i:i+batch_size] = batch_output

    return outputs