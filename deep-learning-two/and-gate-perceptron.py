"""
This is a simple perceptron that acts the same way as a logical AND gate
So it will only "activate" if both of its inputs are 1

AND Truth table
Input 1 | Input 2 | Output
--------------------------
    1   |   1     |     1
    0   |   1     |     0
    1   |   0     |     0
    0   |   0     |     0

Let's see how we can implement this
"""

import random

# First of all we will need the data
data = [[1, 1], [0, 1], [1, 0], [0, 0]]
outputs = [1, 0, 0, 0]
# As our data has two inputs, we will need that many weights + 1 for the bias, so 3
weights = [random.random(), random.random(), random.random()]
# We will also need a bias, I've used a random number but feel free to choose something else like 1
bias = random.random()


# We'll use a really simple activation function in our perceptron called "Heaviside"
def activation_function(some_score):
    return 0 if some_score <= 0 else 1


# Now we can use some matrix math and test it out
def test_perceptron(input_one, input_two, output, weights_vector, bias):
    #  (Sum of[inputs * weights]) + (Bias * weight)
    perceptron_score = \
        input_one * weights_vector[0] \
        + input_two * weights_vector[1] \
        + bias * weights_vector[2]

    result = activation_function(perceptron_score)

    print(f"Actual    : {input_one} & {input_two} -> {output}")
    print(f"Perceptron: {input_one} & {input_two} -> {result}", end=" ")
    if result == output:
        print("Perceptron is correct!")
    else:
        print("Perceptron needs training!")
    return


def test_all(dataset, outputs, weights_vector, bias):
    for X, y in zip(dataset, outputs):
        test_perceptron(X[0], X[1], y, weights_vector, bias)
    return


test_all(data, outputs, weights, bias)

# Okay, so our perceptron is currently not very good, so we need to train it.
learning_rate = 1


# Often the variables will be written like this:
# X is a dataset
# y is the label for a data sample
# W for weights, b for bias
# lr or alpha for learning rate
def train_perceptron(X, y, W, b, alpha=1):
    for data, label in zip(X, y):
        perceptron_score = \
            data[0] * W[0] \
            + data[1] * W[1] \
            + b * W[2]

        result = activation_function(perceptron_score)
        # We only need to adjust the weights if the perceptron is wrong
        error = label - result
        if error:
            W[0] += error * data[0] * alpha
            W[1] += error * data[1] * alpha
            W[2] += error * b * alpha
            print(f"Updated weights to: {W}")

    return W


# Let's train it!
for i in range(10):
    weights = train_perceptron(data, outputs, weights, bias, learning_rate)

# And test it again...
test_all(data, outputs, weights, bias)

# Great! We have now trained a perceptron to act like an AND gate using only some matrix math!
