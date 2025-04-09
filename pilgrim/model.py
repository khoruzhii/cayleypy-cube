import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1, activation_function="relu", use_batch_norm=True):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else None
        self.activation = self._get_activation_function(activation_function)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else None
        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
        out += residual
        out = self.activation(out)
        return out

    @staticmethod
    def _get_activation_function(name):
        if name == "relu":
            return nn.ReLU()
        elif name == "mish":
            return nn.Mish()
        else:
            raise ValueError(f"Unknown activation function: {name}")

class Pilgrim(nn.Module):
    def __init__(self, state_size, hd1=5000, hd2=1000, nrd=2, output_dim=1, dropout_rate=0.1, activation_function="relu", use_batch_norm=True, num_classes=6):
        super(Pilgrim, self).__init__()
        self.dtype = torch.float32
        self.state_size = state_size
        self.num_classes = num_classes
        self.hd1 = hd1
        self.hd2 = hd2
        self.nrd = nrd
        self.use_batch_norm = use_batch_norm
        
        self.input_layer = nn.Linear(state_size * self.num_classes, hd1)
        self.bn1 = nn.BatchNorm1d(hd1) if use_batch_norm else None
        self.activation = self._get_activation_function(activation_function)
        self.dropout = nn.Dropout(dropout_rate)

        if hd2 > 0:
            self.hidden_layer = nn.Linear(hd1, hd2)
            self.bn2 = nn.BatchNorm1d(hd2) if use_batch_norm else None
            hidden_dim_for_output = hd2
        else:
            self.hidden_layer = None
            self.bn2 = None
            hidden_dim_for_output = hd1

        if nrd > 0 and hd2 > 0:
            self.residual_blocks = nn.ModuleList([ResidualBlock(hd2, dropout_rate, activation_function, use_batch_norm) for _ in range(nrd)])
        else:
            self.residual_blocks = None

        self.output_layer = nn.Linear(hidden_dim_for_output, output_dim)

    def forward(self, z):
        x = F.one_hot(z.long(), num_classes=self.num_classes).view(z.size(0), -1).to(self.dtype)
        x = self.input_layer(x)

        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        if self.hidden_layer:
            x = self.hidden_layer(x)
            if self.bn2:
                x = self.bn2(x)
            x = self.activation(x)
            x = self.dropout(x)

        if self.residual_blocks:
            for block in self.residual_blocks:
                x = block(x)

        x = self.output_layer(x)
        return x.flatten()

    @staticmethod
    def _get_activation_function(name):
        if name == "relu":
            return nn.ReLU()
        elif name == "mish":
            return nn.Mish()
        else:
            raise ValueError(f"Unknown activation function: {name}")

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

    outputs = torch.empty(data.size(0), dtype=torch.float16, device=device)

    # Process each batch
    for i in range(0, data.size(0), batch_size):
        batch = data[i:i+batch_size].to(device)
        with torch.no_grad():
            batch_output = model(batch).flatten()
        outputs[i:i+batch_size] = batch_output

    return outputs