"""
No new ground here, just a different dataset!
This will pretty much be the same way I did MNIST digits, perhaps with minor changes

Overview:
Download the dataset
Load it with dataloaders
Create a NN
Train the NN
Test the NN
Show some results

"""

# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms


def view_classify(img, ps):
    """
    This is a function just to give us pretty outputs of an image and what our model believes a digit to be
    https://discuss.pytorch.org/t/view-classify-in-module-helper/30279/6
    """
    ps = ps.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)


batch_size = 32
# Choose some transformations to apply so we can do that as we first process the data
# We'll put the data into tensors and normalise it
fashion_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# (Down)load the data
training_data = datasets.FashionMNIST('FashionMNIST_data/', download=False, train=True, transform=fashion_transforms)
test_data = datasets.FashionMNIST('FashionMNIST_data/', download=False, train=False, transform=fashion_transforms)
# Create some data loaders
training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Let's create a neural network!
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
# Optimiser and a loss_function for the model
optimiser = optim.SGD(fashion_model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

training_epochs = 3
print_every = 500  # Just so we have a manageable amount of text displayed
loss_records = []

for epoch in range(training_epochs):
    running_loss = 0
    print(f"Epoch {epoch + 1} / {training_epochs}")

    for cycle, (images, labels) in enumerate(iter(training_data_loader)):
        # Flatten the images for our network
        images = images.reshape(batch_size, input_size)
        # Zero out the gradients
        optimiser.zero_grad()
        # Make a guess by passing an image forward through the network
        model_guess = fashion_model.forward(images)
        # Calculate how good/bad that guess is
        loss_value = criterion(model_guess, labels)
        # Backpropogate to see how we need to adjust the weights
        loss_value.backward()
        # Adjust the weights
        optimiser.step()

        # Keep track of the loss for our human eyes to see
        running_loss += loss_value.item()

        if cycle % print_every == 0 and cycle != 0:
            print(f"Training cycle: {cycle}\t Avg. Loss: {running_loss / print_every:.4f}")
            running_loss = 0
            loss_records.append([epoch, cycle, loss_value / print_every])

print("=" * 50)
print("DONE")

torch.save(fashion_model.state_dict(), "fashion-model.mod")
#print(len(loss_records))

images, labels = next(iter(test_data_loader))

img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logits = fashion_model.forward(img)

# Softmax on the raw network output to get a "I'm 90% it's this number" output
import torch.nn.functional as F

ps = F.softmax(logits, dim=1)
view_classify(img.view(1, 28, 28), ps)
plt.show()

# Use loss_records to make a visualisation of loss
