import numpy as np # Linear Algebra Library

W = np.array([-2, -2]) # Perceptron weights.
b = 3                  # Perceptron bias.

# Matrix of all possible inputs.
x = np.matrix([[0, 0], [0, 1], [1, 0], [1, 1]])

# Perceptron output for each of the inputs.
np.dot(W, x.transpose()) + b > 0

