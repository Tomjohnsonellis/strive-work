import matplotlib.pyplot as plt
import torch
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random

# import os
# print(os.getcwd())

"""
The goal of this file is to:
Load some pictures of cats and dogs with ImageFolder
--> Apply some transformations as the data is loaded
Feed it into a Network that has some Convolutional Layers
--> Visualise it at each Conv layer





"""
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

# This is a visualisation function, please ignore it
def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.5,0.5,0.5])
        std = np.array([0.5,0.5,0.5])
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
# And another...
class_list = train_data.classes
def view_classify_general(img, ps, class_list):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()
    # print(ps)
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    imshow(img, ax=ax1, normalize=True)
    ax1.axis('off')
    ax2.barh(np.arange(len(class_list)), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(len(class_list)))
    ax2.set_yticklabels([x for x in class_list], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()



def display_train_image():
    # Run this to test your data loaders
    images, labels = next(iter(train_loader))
    print(images.shape)
    imshow(images[0], normalize=True)
    plt.show()
    return

def display_classifier(network):
    random.seed()
    random_index = random.randint(0,20)

    images, labels = next(iter(test_loader))
    img, label = images[random_index], labels[random_index]

    # Forward pass, get our logits
    logits = network(img.view(1, *images[5].shape))
    # logits = network(img.unsqueeze(0))
    # Calculate the loss with the logits and the labels
    print("@"*50)
    print(torch.exp(logits))

    ps = torch.exp(logits)
    view_classify_general(img, ps, class_list)
    



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


def training_loop(model):
    training_epochs = 5
    print_every = 10
    optimiser = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_function = nn.NLLLoss()

    for epoch in range(training_epochs):
        running_loss = 0
        print(f"Epoch {epoch+1} / {training_epochs}")

        for cycle, (images, labels) in enumerate(iter(train_loader)):
            # Flatten the images for our network
            # images = images.reshape(16, 784)
            # print("="*50)
            # print(labels)
            # print("="*50)
            # index = 0
            # for l in labels:
            #     if l == 0:
            #         labels[index] = torch.Tensor([[0][1]])
            #         index += 1
            #     if l == 1:
            #         labels[index] = torch.Tensor([[1][0]])
            #         index += 1


            # Zero out the gradients
            optimiser.zero_grad()
            # Make a guess by passing an image forward through the network
            model_guess = model.forward(images)
            # model_guess = model_guess.topk(1,dim=1)[1].int()
            # Calculate how good/bad that guess is
            # print("-"*50)
            # # print(model_guess.topk(1,dim=1))
            # # model_guess = model_guess.topk(1,dim=1)
            # # model_guess = model_guess[0].flatten()
            # # print(model_guess)
            # print("-"*50)
            loss_value = loss_function(model_guess, labels)
            # print(loss_value)
            # Backpropogate to see how we need to adjust the weights
            loss_value.backward()
            # Adjust the weights
            optimiser.step()

            # Keep track of the loss for our human eyes to see
            running_loss += loss_value.item()

            if cycle % print_every == 0:
                print(f"Training cycle: {cycle}\t Avg. Loss: {running_loss/print_every:.4f}")
                running_loss = 0

    return model


def validate(model):
    model.eval()
    correct = 0
    incorrect = 0
    with torch.no_grad():
        for test_images, test_labels in iter(test_loader):
            # Make a prediction, see if it's correct, keep a record
            model_output = model.forward(test_images)
            probabilities = torch.exp(model_output)
            class_guess = probabilities.topk(1, dim=1)[1].flatten()
            # print(class_guess.shape)
            # print(test_labels.shape)
            scores = test_labels - class_guess
            for score in scores:
                if score == 0:
                    correct += 1
                else:
                    incorrect += 1

            accuracy = (correct / (correct + incorrect)) * 100
            print(f"Currect accuracy: {accuracy}\n-->{correct} / {correct + incorrect}")

    print(f"Final accuracy: {accuracy}")
    model.train()
    return accuracy



if __name__ == "__main__":
    # net = ConvNet()
    # display_classifier(net)
    # After all that, it is time to train the network.
    # trained = training_loop(net)
    # torch.save(trained, "catdog.pt")
    # display_classifier(trained)

    net = torch.load("catdog.pt")
    display_classifier(net)

    # validate(net)



    


