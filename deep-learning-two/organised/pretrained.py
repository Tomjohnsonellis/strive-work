import mymodels
import mydataloaders
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

train_loader, test_loader = mydataloaders.create_catdog_loaders(on_linux=False)
# Alexnet is a ridiculously big model, and we can make use of it!
model = models.alexnet(pretrained=True)
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
model.eval()

# model.classifier will let us see the two most important things, input size and output size
# In this case: in 1024, out 1000
# We will replace the classifier with one that we can use for our more simple dogs/cats problem

# As the model is already trained (contains well-tuned weights and biases) we don't want to make any further adjustments
# model.features.parameters() will display all the weights, so we can freeze gradient calculations with:
for param in model.features.parameters():
    param.requires_grad = False

# Time to replace the classifier, it's actually quite simple: 2 outputs, 1 or 0
model.classifier = nn.Linear(in_features=1024, out_features=2)
# If you like, you can use your own network instead!
model.classifier = mymodels.BinaryClassifier()
print(model)





