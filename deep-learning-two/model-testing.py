import torch
from torch import nn
input_size = 784
hidden_sizes = [128, 64, 32, 16]
output_size = 10

fashion_model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1], hidden_sizes[2]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[2], hidden_sizes[3]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[3], output_size)
)

# model = nn.Sequential(*args, **kwargs)
fashion_model.load_state_dict(torch.load("fashion-model.mod"))
fashion_model.eval()

