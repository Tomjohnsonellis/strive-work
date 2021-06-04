"""

This is a very simple perceptron that is basically just an OR logic gate
It does use weights and biases to update itself

All you need is some matrix multiplication!

"""

import random

learning_rate = 0.1
bias = random.random()
# 3 random weights
weights = [random.random(), random.random(), random.random()]
print(f"Starting weights: {weights}")


def simple_perceptron(input_one, input_two, output):
    # See what the result of the perceptron is
    perceptron_output = input_one * weights[0] + input_two * weights[1] + bias * weights[2]

    # This is a very simple activation function, called "Heaviside"
    if perceptron_output > 0:
        perceptron_output = 1
    else:
        perceptron_output = 0

    # If our perceptron is incorrect, error will be 1 so a change will happen
    # If our perceptron is correct, error will be 0 so nothing happens
    error = output - perceptron_output
    # Update each weight as necessary, the learning rate also determines how much by
    if error:
        weights[0] += error * input_one * learning_rate
        weights[1] += error * input_two * learning_rate
        weights[2] += error * bias * learning_rate
        print(f"Weights updated to: {weights}")


# Let's "train" the perceptron!
for i in range(1000):
    simple_perceptron(1, 1, 1)
    simple_perceptron(1, 0, 1)
    simple_perceptron(0, 1, 1)
    simple_perceptron(0, 0, 0)


# Test cases
def test(x, y):
    perceptron_output = x * weights[0] + y * weights[1] + bias * weights[2]
    if perceptron_output > 0:
        perceptron_output = 1
    else:
        perceptron_output = 0
    print(f"{x} OR {y} is: {perceptron_output}")
    test_results.append(perceptron_output)


test_results = []
test(1, 1)
test(1, 0)
test(0, 1)
test(0, 0)
if test_results == [1, 1, 1, 0]:
    print("All correct!")
else:
    print("Needs more training!")
