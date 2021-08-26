from torch import nn
import torch.nn.functional as F

# This is a model specific for an insurance price task, the data has 10 dimensions
class InsurancePriceNN(nn.Module):
    # Define the layers
    def __init__(self, hidden_sizes):
        super().__init__()
        self.layer1 = nn.Linear(10, hidden_sizes[0])
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.activation2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_sizes[1], 1)
    
    # Define the forward pass
    def forward(self, sample):
        layer1_results = self.layer1(sample)
        layer1_output = self.activation1(layer1_results)
        layer2_results = self.layer2(layer1_output)
        layer2_output = self.activation2(layer2_results)
        network_output = self.layer3(layer2_output)
        return network_output

# A simple test model
class AwfulModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)

    def forward(self, sample):
        network_output = self.layer(sample)
        return network_output

# This model is intended to be used with the cats and dogs data
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            # Image depth, 3 as RGB
            in_channels=3,
            # How many filters are we making?
            out_channels=32,
            # Tweak these as we need
            kernel_size=(5,5),
            # stride=4,
            # padding=2,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(51984, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x, verbose=False):
        if verbose: print(f"Inital input size: {x.shape}")
        x = self.conv1(x)
        if verbose: print(f"Size after conv1: {x.shape}")
        x = F.relu(x)
        x = self.pool(x)
        if verbose: print(f"Size after pool: {x.shape}")
        # Same as above but in more compact code
        # x = self.pool(F.relu(self.conv1(x)))
        x = self.conv2(x)
        if verbose: print(f"Size after conv2: {x.shape}")
        x = F.relu(x)
        x = self.pool(x)
        if verbose: print(f"Size after pool: {x.shape}")


        # A convolutional layer's outputs are a 3D volume,
        # We need it in the form (batch_size, num_filters * kernel_size_x * kernel_size_y)
        # For example: (32, 16 x 5 x 5)
        
        x = x.view(x.shape[0], -1)
        if verbose: print(f"Size after resizing: {x.shape}")
        # We will now give this to the boring old linear neurons
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # And finally put the output in an easier to use form
        x = F.log_softmax(x, dim=1)
        # x = F.softmax(x)
        return x



class SimpleBinaryClassifier(nn.Module):
    def __init__(self, input_size=1024):
        super().__init__()
        self.LayerOne = nn.Linear(input_size, 128)
        self.LayerTwo = nn.Linear(128, 2)

    def forward(self, x):
        x = self.LayerOne(x)
        x = F.relu(x)
        x = self.LayerTwo(x)
        x = F.log_softmax(x, dim=1)
        return x

class DropoutNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.FullCon1 = nn.Linear(1024, 128)
        self.FullCon2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.FullCon1)
        x = F.dropout(x)
        x = self.FullCon2(x)
        x = F.dropout(x)
        x = F.log_softmax(x, dim=1)
        return x

egg = SimpleBinaryClassifier()