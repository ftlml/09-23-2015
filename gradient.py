from sklearn.datasets.samples_generator import make_regression # Sample Data Gen
import matplotlib.pyplot as plt # Plotting Library

# Sample data.
x, y = make_regression(n_samples = 100, n_features = 1, n_informative = 1, 
                       random_state = 0, noise = 35)
# Initialize the parameters.
theta0 = 0
theta1 = 0
# Start Gradient Descent.
alpha = 0.01
n_iterations = 500
n_samples = len(x)
for _ in xrange(n_iterations):
  # Compute the derivative of the cost function
  # with respect to theta0.
  theta0_grad = 1.0 / n_samples * sum([(theta0 + theta1 * x[idx] - y[idx]) \
                                       for idx in xrange(n_samples)])
  # Compute the derivative of the cost function
  # with respect to theta1.
  theta1_grad = 1.0 / n_samples * sum([(theta0 + theta1 * x[idx] - y[idx]) * x[idx] \
                                       for idx in range(n_samples)])
  # Update the parameters.
  theta0 = theta0 - alpha * theta0_grad
  theta1 = theta1 - alpha * theta1_grad
# Plot the results.
for _ in xrange(x.shape[1]):
  predictions = theta0 + theta1 * x
plt.plot(x, y, 'o')
plt.plot(x, predictions, 'k-')
plt.show()