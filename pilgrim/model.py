# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


class Pilgrim(nn.Module):
    def __init__(self, state_size, hd1=5000, hd2=1000, nrd=2, output_dim=1, dropout_rate=0.0, num_classes=6):
        super(Pilgrim, self).__init__()
        self.dtype = torch.float32
        self.state_size = state_size
        self.num_classes = num_classes
        self.hd1 = hd1
        self.hd2 = hd2
        self.nrd = nrd
        self.z_add = 0

        self.input_layer = nn.Linear(state_size * num_classes, hd1)
        self.bn1 = nn.BatchNorm1d(hd1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        if hd2 > 0:
            self.hidden_layer = nn.Linear(hd1, hd2)
            self.bn2 = nn.BatchNorm1d(hd2)
            hidden_dim_for_output = hd2
        else:
            self.hidden_layer = None
            self.bn2 = None
            hidden_dim_for_output = hd1

        if nrd > 0 and hd2 > 0:
            self.residual_blocks = nn.ModuleList(
                [ResidualBlock(hd2, dropout_rate) for _ in range(nrd)]
            )
        else:
            self.residual_blocks = None

        self.output_layer = nn.Linear(hidden_dim_for_output, output_dim)

    def forward(self, z):
        # One-hot encode and flatten to dense
        x = F.one_hot(z.long() + self.z_add, num_classes=self.num_classes).view(z.size(0), -1).to(self.dtype)

        # Input block
        x = self.input_layer(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Optional hidden block
        if self.hidden_layer is not None:
            x = self.hidden_layer(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.dropout(x)

        # Optional residual stack
        if self.residual_blocks is not None:
            for block in self.residual_blocks:
                x = block(x)

        # Output
        x = self.output_layer(x)
        return x.flatten()


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

    outputs = torch.empty(data.size(0), dtype=torch.float16, device=device)

    # Process each batch
    for i in range(0, data.size(0), batch_size):
        batch = data[i:i + batch_size].to(device)
        with torch.no_grad():
            batch_output = model(batch).flatten()
        outputs[i:i + batch_size] = batch_output

    return outputs
