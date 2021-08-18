"""
This was just for debugging, but it could be salvaged for a simple training loop or checked for reference
"""


import torch
import numpy as np
import pandas as pd
import torch.nn as nn

data = pd.read_csv("work-in-progress/data.csv", header=None)
X = data.drop(2, axis=1).values
y = data[2].values
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()
y = y.view(100, 1)

class A_Simple_Neural_Network_With_A_Forward_Pass(nn.Module):
    def __init__(self, input_dimensions, hidden_layer_size, output_size=1):
        super().__init__()
        self.first_linear_calculation = nn.Linear(input_dimensions, hidden_layer_size)
        self.sigmoid = nn.Sigmoid()
        self.second_linear_calculation = nn.Linear(hidden_layer_size, output_size)

    def forward(self, data_given_to_the_network):
        results_of_first_linear_calculation = self.first_linear_calculation(data_given_to_the_network)
        activation_results = self.sigmoid(results_of_first_linear_calculation)
        results_of_second_linear_calculation = self.second_linear_calculation(activation_results)
        network_output = self.sigmoid(results_of_second_linear_calculation)
        return network_output

def train_our_simple_network(data, labels, our_model, some_loss_function, some_learning_rate, number_of_epochs):
    # Typical practice is to define the optimiser inside the training function, as that is where it will be used
    optimiser = torch.optim.Adam(our_model.parameters(), lr=some_learning_rate)
    for epoch in range(number_of_epochs):
        optimiser.zero_grad() # This just ensures that we recalculate the gradients from scratch each epoch
        label_predictions = our_model(data) # Make some predictions
        loss_value = some_loss_function(label_predictions, labels)  # Calculate how good they were

        print(f"Epoch {epoch + 1}: Loss = {loss_value}")

        loss_value.backward() # Adjust the network accordingly
        optimiser.step() # Updates our parameters
    return our_model


simple_net = A_Simple_Neural_Network_With_A_Forward_Pass(2, 10, 1)
trained_network = train_our_simple_network(X, y, simple_net, nn.BCELoss(), 0.25, 10)