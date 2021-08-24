"""
In the interest of making sure I get the logic for NNs correct,
This is a perceptron that functions like an OR gate
It should activate if either of the inputs are 1

OR Truth table
Input 1 | Input 2 | Output
--------------------------
    1   |   1     |     1
    0   |   1     |     1
    1   |   0     |     1
    0   |   0     |     0

"""
import random

weights = [random.random(), random.random()]
bias = random.random()
# print(f"Starting weights: {weights}")
# print(f"Starting bias: {bias}")

data = [[1, 1], [0, 1], [1, 0], [0, 0]]
outputs = [1, 1, 1, 0]
dataset = zip(data, outputs)


# We'll use a really simple activation function in our perceptron called "Heaviside"
def activation_function(some_score):
    return 0 if some_score <= 0 else 1


def perceptron(X, y, W, b, lr=0.1):
    score = X[0] * W[0] + X[1] * W[1] + b
    prediction = activation_function(score)
    error = y - prediction

    print(f"Actual: {y} ~~ Prediction: {prediction}")
    if error:
        print(f"Error: {error}")
        print("Adjusting weights and bias...")
        W[0] += X[0] * error * lr
        W[1] += X[1] * error * lr
        b += error * lr
        print(f"Weights adjusted to: {W}")
        print(f"Bias adjusted to {b}")
    else:
        print("Ok")

    return W, b


# Have a look at the data
for X, y in zip(data, outputs):
    print(X, y)

# See how the perceptron handles the data

for e in range(10):
    print(f"Epoch: {e+1}")
    dataset = zip(data, outputs)
    for X, y in dataset:
        weights, bias = perceptron(X, y, weights, bias, lr=1)

print("="*50)
print(f"Final weights: {weights}")
print(f"Final bias: {bias}")