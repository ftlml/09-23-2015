import cPickle # Python Serialization
import gzip # GZip Library
import matplotlib.pyplot as plt
import numpy as np # Linear Algebra Library
import theano
import theano.tensor as T
import time

class SigmoidLayer:
  def __init__(self, X, n_in, n_out):
    # Initialize network parameters.
    W = theano.shared(np.random.randn(n_in, n_out))
    b = theano.shared(np.zeros(n_out))
    self.params = [W, b]
    # Compute layer activations
    self.output = T.nnet.sigmoid(theano.dot(X, W) + b)

class SoftmaxLayer:
  def __init__(self, X, n_in, n_out):
    W = theano.shared(np.zeros((n_in, n_out)))
    b = theano.shared(np.zeros(n_out))
    self.params = [W, b]
    # Compute layer activations
    self.output = T.nnet.softmax(T.dot(X, W) + b)

class NeuralNetwork:
  def __init__(self, n_in, n_hidden, n_out, lr = .2):
    lr = theano.shared(np.cast[theano.config.floatX](lr))
    X = T.matrix('X')
    Y = T.ivector('Y')
    # Create layers.
    layer_1 = SigmoidLayer(X, n_in, n_hidden)
    layer_2 = SoftmaxLayer(layer_1.output, n_hidden, n_out)
    # Negative log likelihood cost function.
    cost = -T.sum(T.log(layer_2.output)[T.arange(Y.shape[0]), Y])
    # Computes the network parameter updates.
    updates = []
    for layer in [layer_1, layer_2]:
      for param in layer.params:
        gradient = T.grad(cost, param)
        updates.append((param, param - lr * gradient))
    # Backprop the error through the network.
    self.train = theano.function(inputs = [X, Y],
                                 outputs = cost,
                                 updates = updates,
                                 allow_input_downcast = True)
    # Make a prediction.
    self.predict = theano.function(inputs = [X], outputs = layer_2.output)

# Load the dataset
input = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(input)
input.close()
# Define runtime parameters.
batch_sz = 10
n_batches = len(train_set[0]) / batch_sz
n_epochs = 100
# Train a MLP network.
nn = NeuralNetwork(784, 100, 10)
train_errors = np.ndarray(n_epochs)
train_start = time.time()
for epoch in xrange(n_epochs):
  error = 0.
  for batch in xrange(n_batches):
    start = batch * batch_sz
    end = start + batch_sz
    error += nn.train(train_set[0][start:end], train_set[1][start:end])
  train_errors[epoch] = error
  print 'Epoch [%i] Cost: %f' % (epoch, error)
train_end = time.time()
print
print 'Training Time: %g seconds' % (train_end - train_start)
print
# Plot the learning curve.
plt.plot(np.arange(n_epochs), train_errors, 'b-')
plt.xlabel('epochs')
plt.ylabel('error')
plt.show()
# Use test set to determine accuracy.
test_x, test_y = test_set
def predict(x):
  return np.argmax(nn.predict([x]))
predictions = [(predict(x), y) for (x, y) in zip(test_x, test_y)]
correct = sum(int(x == y) for (x, y) in predictions)
print 'Correctly predicted %i of %i for %i%% accuracy.' % \
      (correct, len(test_x), float(correct) / len(test_x) * 100)
print