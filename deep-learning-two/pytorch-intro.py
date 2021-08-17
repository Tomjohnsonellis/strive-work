import torch
import torch.nn as nn
import numpy as np

print("=== 1")
# You can manually create tensors like this:
X = torch.tensor(([1, 1], [2, 2], [3, 3], [4, 4], [5, 5]), dtype=torch.int32)
print(X.size())  # 5 tensors of dimension 2
y = torch.tensor(([11], [22], [33], [44], [55]), dtype=torch.int32)
print(y.size())  # 5 tensors of dimension 1
test_sample = torch.tensor(([3, 3]), dtype=torch.int32)

print("=== 2")
# Or you can convert from numpy arrays
big_array = np.random.normal(size=[100, 10])
print(type(big_array))
big_tensor = torch.from_numpy(big_array)
print(type(big_tensor))

print("=== 3")


# Neural Networks are commonly used with pytorch and we can create one like this:
class My_Barebones_Neural_Network(nn.Module):
    def __init__(self):
        super(My_Barebones_Neural_Network, self).__init__()


# And make it somewhat useful by adding number of inputs, outputs and the size of a hidden layer
class My_Neural_Network_With_Some_Parameters(nn.Module):
    def __init__(self, input_dimensions, size_of_hidden_dimensions, output_size):
        super(My_Neural_Network_With_Some_Parameters).__init__()


# And we can make a simple, fully connected neural network which uses the Sigmoid function as its activation like this:
class A_Simple_Neural_Network(nn.Module):
    def __init__(self, input_dimensions, hidden_layer_size, output_size=1):
        super().__init__()
        self.first_linear_calculation = nn.Linear(input_dimensions, hidden_layer_size)
        self.sigmoid = nn.Sigmoid()
        self.second_linear_calculation = nn.Linear(hidden_layer_size, output_size)


# As this is a class, we can instantiate it as we need to!
my_net = A_Simple_Neural_Network(input_dimensions=10, hidden_layer_size=5)


# With Pytorch, we do need to define our own forward pass, which is just how we want our network to work structurally
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


# Great! We've made a simple neural network class that we can use!
# Let's try it out on some junk data
sample_size = 10
sample = torch.from_numpy(np.array(np.random.rand(sample_size), dtype=np.float32))
print(f"Sample: {sample}")
# Let's instantiate another network
simple_net = A_Simple_Neural_Network_With_A_Forward_Pass(input_dimensions=sample_size, hidden_layer_size=5)
# And give it our sample
result = simple_net.forward(sample)
print(f"Result {result[0]}")
# Nice, it works! It returns a value between 0 and 1 as all it does is take some values and sigmoid them
# Next up is the optimisation, loss function and backpropagation, oh my.
# For optimisation we need to choose a loss function that makes sense for our task
# https://pytorch.org/docs/stable/nn.html#loss-functions We're making a binary classifier, so we'll go with BinaryCrossEntropy
loss_function = nn.BCELoss()
# And then we need some way to *optimise* the loss
# https://pytorch.org/docs/stable/optim.html
# We're going to use Adam, for an explanation of what it is, check out:
# https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
optimiser = torch.optim.Adam(simple_net.parameters(), lr=0.1)


