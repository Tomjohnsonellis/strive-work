"""
Perceptron: Used for supervised learning of binary classifiers
The simplest form of a neural network, it is used for "yes/no" type problems
For more info: https://towardsdatascience.com/what-the-hell-is-perceptron-626217814f53
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# A perceptron requires:
# Some data to learn from
# Weights and Biases
# An activation function (like Sigmoid or ReLu)
#
# Let's make a perceptron to help us choose a movie to watch, for this example we like romance and don't like Marvel
#
# Data
# The sample films are Titanic, Avengers: Endgame, I still believe, Spiderman
# Is it a Marvel film?
x1 = np.array([0, 1, 0, 1, ])
# Is it Romantic?
x2 = np.array([1, 0, 1, 0, ])
# Has it won an Oscar?
x3 = np.array([1, 0, 0, 0, ])
# Did we enjoy watching it? (This is our prediction target)
y = np.array([1, 0, 1, 0, ])
# Weights and biases
# Let's use some random values at first, we will need as many weights as we have data points, 3 in this case
w1, w2, w3 = np.random.rand(3)
print(f"Starting Weights: {w1, w2, w3}")
# We'll also need a bias
b = np.random.rand(1)[0]

# Just for fun, we'll make one of the weights really bad
w2 = -2 * w2

# We'll use the perceptron to guess if we will like another movie, a romantic oscar winning film called "Chocolat"
x1_test = 0
x2_test = 1
x3_test = 1

# Now that we have set up all the needed variables we can start training the perceptron!
# First off, let's see if our random weights are accurate for the first film (Titanic)
# Sum Of(Data x Weights) + Bias
# Mathematically: x1*w1 + x2*w2 + x3*w3 + b
titanic = (x1[0] * w1) + (x2[0] * w2) + (x3[0] * w3) + b
print(f'Titanic\'s score: {titanic}')


# This number is neither 0 or 1, but we can use an activation function to map the result to one of those values!
# We will use a very basic one here, if the result was positive we map it to 1, otherwise we'll map it to 0
def activation(result):
    if result >= 0:
        return 1
    elif result <= 0:
        return 0

print(f"Did we enjoy this film?: {activation(titanic)}")

# This is incorrect, our data states that we did in fact enjoy Titanic, so let's adjust our weights!
# We do this by changing the weights based on how incorrect we were, bigger errors would need bigger adjustments!
# Mathemetically: w(new) = w(old) + (error * data)

# As we are doing a binary classification, we either have an error or don't
error = np.array(y[0] - activation(titanic))
w1 += error * x1[0]
w2 += error * x2[0]
w3 += error * x3[0]

# Let's try another prediction after these adjustments
print("=== Weights have been adjusted ===")
titanic = (x1[0] * w1) + (x2[0] * w2) + (x3[0] * w3) + b
print(f'Titanic\'s score: {titanic}')
print(f"Did we enjoy this film?: {activation(titanic)}")
# Great! We have corrected our perceptron on this issue!
# These adjustments are very large though
# We will introduce a learning rate so that we can make much finer adjustments
learning_rate = 0.1
# Applying it to the formula gives us
# w(new) = w(old) + (error * data)