import os
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
"""
This will be an example of using a CNN to classify cats and dogs
Interesting things to note will be how the CNN is structured
"""




print(os.getcwd())
data_dir = "../datasets/cats_and_dogs"

# Define some transformations for the data
train_transforms = transforms.Compose([transforms.Resize(255),
transforms.RandomRotation(15),
transforms.RandomResizedCrop(240),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
])


# Resize, Center Crop, Tensor Normalize
test_transforms = transforms.Compose([transforms.Resize(255),
transforms.CenterCrop(240), 
transforms.ToTensor(),
transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
])

# Create loaders
train_data = datasets.ImageFolder(data_dir + '/train', transform = train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

# This is a helper function for visualisation, not the focus here
def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax



train_loader = DataLoader(dataset=train_data, batch_size=32, drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=32, drop_last=True)
    
# To display an image, use:
# images, labels = next(iter(train_loader))
# imshow(images[0], normalize=False)
# plt.show()

# Time to create a model
import torch.nn as nn
import torch.nn.functional as F


"""
The challenge of model building with CNNs is 
getting all the output sizes correct.
The general formula is:
(input_size - kernal_size + (2*padding) / stride) + 1
"""
def calc_output_size(input_size, kernel_size, padding, stride):
    output = (input_size - kernel_size + 2*padding / stride) + 1
    return output




class CDNet(nn.Module):
    def __init__(self):
        super().__init__
        self.first_conv = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(3,3)
        self.second_conv = nn.Conv2d(32, 16, 5)
        self.fc1 = nn.Linear(16*53*53, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,2)

    def forward(self, x):
        pass




if __name__ == '__main__':
    image = iter(test_loader).next()
    print(image[0])
    # image=image.squeeze(0)
    # plt.imshow(image)

