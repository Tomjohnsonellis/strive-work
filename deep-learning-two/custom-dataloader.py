"""
Given a dataset that has data & labels, can we make it easier to handle?
Something helpful for loading data into a neural network is a data loader!
"""

#imports
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math

def split_section(section_number):
    print("="*10, end=" ")
    print(f"Section {section_number}", end=" ")
    print("="*20)



# Here's a trivial dataset for this example:
X = [1,2,3,4,5]
y = [0,0,0,1,1]
dataset = X, y

# We'll oop this
class CustomDataset(Dataset):
    pass

# We want to be able to index the data by using dataset[i]
# To do this we will write a new __getitem__ method
# We will also need to replace the constructor (__init__) and
# We will make a new __len__ that is more useful for us
class CustomDataset(Dataset):
    def __init__(self, csv_file): # We'll be using .csv for this example
        pass
    def __getitem__(self, index):
        pass
    def __len__(self):
        pass


# We'll use some house price data
dataset_path = "https://people.sc.fsu.edu/~jburkardt/data/csv/homes.csv"
df = pd.read_csv(dataset_path)
# You can take a look if you like!
#print(df)
# Here's the info we care about
split_section(1)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns}")
df.columns = [x.replace('"', '').replace(' ', '') for x in df.columns]
print(df.columns)

# We can do some preprocessing inside the data loader!
# We'll also exclude any houses above a budget level
class HouseDataset(Dataset):
    def __init__(self, csv_file, budget=150):
        df = pd.read_csv(csv_file) # Load in the data
        # The dataset has some extra quotes and spaces in the column names 
        # Let's remove them!
        df.columns = [x.replace('"', '').replace(' ', '') for x in df.columns]
        # And let's take just some of the columns, perhaps the most important ones
        important_columns = ["Rooms", "Beds", "Baths", "Age", "Acres", "Taxes"]
        self.X = df[important_columns].values # A nice array to work with
        # Remove houses out of budget
        self.y = (df.Sell.values <= budget).astype("int")
        # As we are excluding some data, the number of samples will have changed
        # So we will use the new length of our new X/data 
        self.n_samples = len(self.X)
    def __len__(self):
        # Using the new value that was created in the constructor...
        return self.n_samples
    def __getitem__(self, index):
       # To do our preferred way of indexing (dataset[index])
       # We just make a method to do it for us, saves us some hassle
       return self.X[index], self.y[index]

# Great! That should be all we need, let's try it...
# dataset_path = "https://people.sc.fsu.edu/~jburkardt/data/csv/homes.csv"
low_budget_dataset = HouseDataset(dataset_path, budget=100)
# Pass this into pytorch's DataLoader utility...
data_loader = DataLoader(dataset=low_budget_dataset, batch_size=4, shuffle=True)
# Make an iterator for it...
data_iterator = iter(data_loader)
# Annnnd grab some data!
some_data = data_iterator.next()
split_section(2)
print(some_data)
# Great! it gives us a batch of 4 samples, with both the data(X) and label(y)


"""
Image Data
If you are working with images, pytorch has something similar to DataLoader
that is made specifically for image datasets, ImageFolder
(part of the "torchvision datasets" module)

Note: Your data must be structured like this...
- root directory for training data
|- class_one
    |- image_one.png
    |- image_two.png
|- class_two
    |- image_one.png
    |- image_two.png

(You'll need a similar structure for the test data)
"""
from torchvision import datasets, transforms
# I have my data in a different place, this is just so I can access it
import os
print(os.getcwd())
os.chdir("../../vscode/datasets")
root_path = "cats_and_dogs"

# A quick refresher on transformations...
# With image data, you can get improved results by
# applying a few simple effects to mess up the images slightly,
# Rotate them, flip them, crop, etc.
train_transforms = transforms.Compose([transforms.Resize(255), # Standardise the size
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])])
# We don't need to apply any alterations to the test data other than a crop,
# And that's just because we cropped the training data
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5],
                                                           [0.5, 0.5, 0.5])])

# ImageFolder behaves similarly to DataLoader, we can load images and transform them
train_data = datasets.ImageFolder(root_path + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(root_path + '/test', transform=test_transforms)

from torch.utils.data import DataLoader # Put the images into loaders
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
test_loader = DataLoader(test_data, batch_size=5, shuffle=True)
# The default loader will do for this task

"""
Below is a helpful function that allows us to visualise an image
"""
import matplotlib.pyplot as plt
def imshow(image, ax=None, title=None, normalize=False):
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


# Run this to load in the images
images, labels = next(iter(train_loader))
# And this will bring up an image
imshow(images[0], normalize=False)
split_section(3)
print(">Displaying Image")
plt.show()

"""
Disorganised Data

It was mentioned earlier that your data needed to be in a specific structure,
While it is best practice to keep data well organised,
Sometimes it gets messy, good thing we can build our own data loaders!

We can use filepaths for each individual image if we need to!
An example of a .csv :
        path                label
0   my_dataset/image1.png       0
1   my_dataset/image2.png       1
"""

