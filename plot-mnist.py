import cPickle # Python Serialization
import gzip # GZip Library
import matplotlib.pyplot as plt # Plotting Library
import matplotlib.cm as cm # Color Maps

# Load the dataset
input = gzip.open('Data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(input)
input.close()
# Expand the training set to inputs and target lebels
# then select a sample to display.
x, y = train_set
sample = 0
# Print the sample label to console.
print 'Label: %i' % y[sample]
# Plot the sample input.
image = x[sample].reshape((28, 28))
plt.imshow(image, cmap = cm.Greys_r)
plt.show()