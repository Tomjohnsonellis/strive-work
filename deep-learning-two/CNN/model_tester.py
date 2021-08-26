import torch
from cats_and_dogs import display_classifier, ConvNet, validate, training_loop

def make_a_prediction(model) -> None:
    display_classifier(model)
    return


def train_further(model) -> ConvNet:
    training_loop(model)


if __name__ == '__main__':
    model = torch.load("catdog.pt")
    train_further(model)
    make_a_prediction(model)

    # validate(model)