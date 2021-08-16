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
# w(new) = w(old) + (learning_rate * error * data)


# What now?
# We will now go through our dataset and tweak the weights as needed for each sample
# Then we will do that all over again and repeat until the perceptron is always right
# After that we can test it on some new data and see what its prediction is!

# Let's compact what we did
learning_rate = 0.2
# this is just the same data as above in a single data structure, outcomes are the digit outside the array
data = [
    (np.array([0,1,1]), 1),
    (np.array([1,0,0]), 1),
    (np.array([0,1,0]), 0),
    (np.array([1,0,0]), 1),
    (np.array([0,1,1]), 1),
]
# Train on everything but the last sample
training_data = data[:-1]
# The last sample is our test
test_data = data[-1]
# Initialise weights at zero or random
weights = np.zeros(3)
weights = np.random.rand(3)
bias = 0
# To prevent endless training
max_iterations = 10

def perceptron(train, weights, bias, learning_rate, max_iterations):
    current_iteration = 0
    print("=== PERCEPTRON ===")
    print(f"Current weights: {weights}")
    print(f"Current bias: {bias}")
    while current_iteration < max_iterations:
        new_weights = weights
        new_bias = bias
        for sample in train:
            # sample[0] is the data, sample[1] is the outcome
            output = np.dot(weights, sample[0]) + bias
            output = activation(output) # Make it 0 or 1

            if (sample[1] - output) != 0: # If we have an error
                # Update the weights and bias
                weights += learning_rate * (sample[1] - output)*sample[0]
                bias += learning_rate * (sample[1] - output)
                print(f"Weights updated to: {weights}")
                print(f"Bias updated to: {bias}")

        # We can check to see if we have converged
        if new_weights.all() == weights.all() and new_bias == bias:
            print(f"We have converged! Iteration {current_iteration+1}")
            break
        current_iteration += 1

    return weights, bias

# To see this in action...
weights, bias = perceptron(training_data, weights, bias, learning_rate, max_iterations)
movie_titles = ['Titanic', 'Avengers: Endgame', 'I still believe', 'Spiderman']
for i, sample in enumerate(training_data):
    output = np.dot(weights, sample[0]) + bias
    print(movie_titles[i], "prediction:", int(output>0), "actual:", sample[1])

# Nice! Now we can finally use it to predict if we would like a movie...
chocolat = np.array([1,0,1])
output = np.dot(weights, chocolat) + bias
print(f"The perceptron's prediction for us enjoying the film is: {output>0}")
