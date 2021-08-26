import torch
from torchvision import transforms, datasets

# normalisation_params = [[0.5,0.5,0.5],[0.5,0.5,0.5]]

# training_transformations = transforms.Compose([
#     transforms.Resize(255),
#     transforms.RandomRotation(20), 
#     transforms.RandomResizedCrop(240),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(*normalisation_params),
# ])

# testing_transformations = transforms.Compose([
#     # We have no need to flip or rotate the test images
#     transforms.Resize(255),
#     transforms.CenterCrop(240),
#     transforms.ToTensor(),
#     transforms.Normalize(*normalisation_params),
# ])

# data_dir = ("../datasets/cats_and_dogs") # Linux
# data_dir = "../../vscode/datasets/cats_and_dogs" # Windows
# train_data = datasets.ImageFolder(data_dir + "/train", transform=training_transformations)
# test_data = datasets.ImageFolder(data_dir + "/test", transform=testing_transformations)

# train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

def create_catdog_loaders(batch_size=32, on_linux=True):

    normalisation_params = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

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


    if on_linux:
        data_dir = ("../datasets/cats_and_dogs") # Linux
    else:
        data_dir = "../../vscode/datasets/cats_and_dogs" # Windows

    train_data = datasets.ImageFolder(data_dir + "/train", transform=training_transformations)
    print(train_data.class_to_idx)
    test_data = datasets.ImageFolder(data_dir + "/test", transform=testing_transformations)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

if __name__ == '__main__':
    print("create_catdog_loaders(batch_size=32, on_linux=True)")
    # x, y = create_catdog_loaders(16, False)

