import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet


# We will be training a model to predict the category, given a sentence.
# Data Structure:
# {
#     intents[
#         {
#             tag:"Greetings",
#             patterns: ["hi","hey"],
#             responses: ["yo","yes?"]
#         }
#     ]
# }
# An "intent" is a tag, pattern and response


# We will split the data into categories
def seperate_data(intents_dict:dict) -> tuple[list,list,list] :
    # Every unique word
    all_words = []
    # Every category
    tags = []
    # Every 
    patterns = []


    for intent in intents_dict['intents']: # The whole dataset
        # Isolate the tag section
        tag = intent['tag']
        # Add it to the categories
        tags.append(tag)

        # Then for each of the samples of that tag
        for pattern in intent['patterns']:

            words = tokenize(pattern)
            # Extend inserts each item rather than appending a list
            all_words.extend(words)
            # Patterns will be the X and y
            patterns.append((words, tag))

    return all_words, tags, patterns

def augment_data(all_words, tags) -> tuple[list,list]:
    # We'll use stemming to increase the size of the training data
    # Stemming is the opposite of Lemmatisation (running -> To run)
    
    # We do not care for punctuation
    ignore_words = ["!","?","."]
    all_words = [stem(word) for word in all_words if word not in ignore_words]
    # No need for duplicates
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    return all_words, tags


def create_training_data(all_words, tags, patterns):
    X_train = []
    Y_train = []

    # Create a "bag of words" for each of the patterns we have
    for pattern_sentence, tag in patterns:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        label = tags.index(tag)
        Y_train.append(label)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    return X_train, Y_train


class ChatData(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.n_samples = len(X)
        self.X = X
        self.y = y
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.n_samples


def train_model(dataset, tags):
    # Hyper Parameters
    num_epoches = 1000
    batch_size = 16
    learning_rate = 0.001
    input_size = len(dataset[0][0])
    hidden_size = 16
    output_size = len(tags)

    # Just in case you have a GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)


    # Standard NN Training loop            
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epoches):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            outputs = model(words)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if(epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epoches}], Loss: {loss.item():.4f}')

    print(f'final loss: {loss.item():.4f}')

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }

    file = "natural-language-processing/chatbot/trained.pth"
    torch.save(data, file)

    print(f'training complete. file saved!')
    return model




    


if __name__ == "__main__":
    # Load our training data, which in this case is a dictionary of sentences.
    # Each sentence is part of a category like "greetings", "thanks" or "payments"
    with open('natural-language-processing/chatbot/intents.json', 'r') as f:
        intents = json.load(f)

    # Collect the data
    all_words, tags, patterns = seperate_data(intents)
    all_words, tags = augment_data(all_words, tags)
    # Put it into a dataset
    X, y = create_training_data(all_words, tags, patterns)

    dataset = ChatData(X, y)

    trained_model = train_model(dataset, tags)