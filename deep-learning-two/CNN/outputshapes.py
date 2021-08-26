import torch
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F





# We want our normalisation to be the same for both train and test
normalisation_params = [[0.5,0.5,0.5],[0.5,0.5,0.5]]

training_transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.RandomRotation(20), 
    transforms.RandomResizedCrop(240),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*normalisation_params),
])

testing_transformations = transforms.Compose([
# We have no need to flip or rotate the test images
    transforms.Resize(255),
    transforms.CenterCrop(240),
    transforms.ToTensor(),
    transforms.Normalize(*normalisation_params),
])

data_dir = ("../datasets/cats_and_dogs")
train_data = datasets.ImageFolder(data_dir + "/train", transform=training_transformations)
test_data = datasets.ImageFolder(data_dir + "/test", transform=testing_transformations)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)


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
        self.fc1 = nn.Linear(16*53*53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        # Same as above but in more compact code
        x = self.pool(F.relu(self.conv2(x)))
        # A convolutional layer's outputs are a 3D volume,
        # We need it in the form (batch_size, num_filters * kernel_size_x * kernel_size_y)
        # So for this specific case, (32, 16 x 5 x 5)
        x = x.view(x.shape[0], -1)
        # We will now give this to the boring old linear neurons
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # And finally put the output in an easier to use form
        x = F.log_softmax(x, dim=1)


images, labels = next(iter(train_loader))
print(f"Batch shape: {images.shape}")
# print(images[0].shape)
one_batch_of_images = images

conv1 = nn.Conv2d(in_channels=3,out_channels=50,kernel_size=(5,5))
pool = nn.MaxPool2d(kernel_size=2, stride=2)
conv2 = nn.Conv2d(in_channels=50, out_channels=25, kernel_size=5)
fc1 = nn.Linear(16*53*53, 120) # This will actually need redefining!
fc2 = nn.Linear(120, 84)
fc3 = nn.Linear(84, 2)

print("Passing through first conv layer...")
one_batch_of_images = conv1(one_batch_of_images)
print(f"After conv1, the shape is now: {one_batch_of_images.shape}")
print("Applying ReLU...")
one_batch_of_images = F.relu(one_batch_of_images)
print(f"After ReLU, the shape is now: {one_batch_of_images.shape}")
print("Applying max pool...")
one_batch_of_images = pool(one_batch_of_images)
print(f"After max pool, the shape is now: {one_batch_of_images.shape}")
print("Passing through second conv layer...")
one_batch_of_images = conv2(one_batch_of_images)
print(f"After conv2, the shape is now: {one_batch_of_images.shape}")
print("Applying ReLU...")
one_batch_of_images = F.relu(one_batch_of_images)
print(f"After ReLU, the shape is now: {one_batch_of_images.shape}")
print("Applying max pool...")
one_batch_of_images = pool(one_batch_of_images)
print(f"After max pool, the shape is now: {one_batch_of_images.shape}")
#####
print("Reshaping to feed into linear section...")
# This gives us (batch_size, total_parameters) as the shape
one_batch_of_images = one_batch_of_images.view(one_batch_of_images.shape[0], -1)
print(f"After the view command, the shape is now: {one_batch_of_images.shape}")
# For this example, the second conv layer output size (25 x 57 x 57) totals to 81225 parameters!
# Which we will now give to the linear network
fc1 = nn.Linear(81225, 120) 
fc2 = nn.Linear(120, 84)
fc3 = nn.Linear(84, 2)
print("Giving to fully-connected layer 1...")
one_batch_of_images = fc1(one_batch_of_images)
print(f"After fc1, the shape is now: {one_batch_of_images.shape}")
print("Passing through fc2...")
one_batch_of_images = fc2(one_batch_of_images)
print(f"After fc2, the shape is now: {one_batch_of_images.shape}")
print("Passing through fc3...")
one_batch_of_images = fc3(one_batch_of_images)
print(f"After fc3, the shape is now: {one_batch_of_images.shape}")
# Annnnnd finally, softmax those values to get our final predictions
print("Applying softmax...")
one_batch_of_images = F.softmax(one_batch_of_images, dim=1)
print(f"After softmax, the shape is now: {one_batch_of_images.shape}")
print(one_batch_of_images)

# # Let's sigmoid the outputs
# one_batch_of_images = one_batch_of_images.sigmoid()




print("The images have been turned into our model's predictions for each class!")
print(f"Example: {one_batch_of_images[5]}")
# We only want the most confident prediction for loss
# one_batch_of_images = one_batch_of_images.topk(1, dim=1)
# print(one_batch_of_images[1])


"""


This will hopefully allow someone to follow throught the shape changes in a network


"""



