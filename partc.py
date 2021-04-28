import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
class Adaline(object):

    def __init__(self, eta=0.0001, epochs=1000):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):

        self.w_ =  np.random.normal(0, 0.0001, 1 + X.shape[1])

        for i in range(self.epochs):
            for xi, target in zip(X, y):
                output = self.net_input(xi)
                error = (target - output)
                self.w_[1:] += self.eta * xi*error
                self.w_[0] += self.eta * error

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) > 0.0, 1, -1)


X = np.load("train21.npy")
y =  np.load("lables21.npy")
test_data = np.load("test_data_b.npy")
test_labels = np.load("test_lables_b.npy")

ada = Adaline()
ada.train(X, y)
print(ada.predict(X))
print(ada.w_)
def plot_decision_regions(X, y, classifier, resolution=0.02):
   # setup marker generator and color map
   markers = ('s', 'x', 'o', '^', 'v')
   colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
   cmap = ListedColormap(colors[:len(np.unique(y))])

   # plot the decision surface
   x1_min, x1_max = X[:,  0].min() - 1, X[:, 0].max() + 1
   x2_min, x2_max =  X[:, 1].min() - 1, X[:, 1].max() + 1
   xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
   np.arange(x2_min, x2_max, resolution))
   Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
   Z = Z.reshape(xx1.shape)
   plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
   plt.xlim(xx1.min(), xx1.max())
   plt.ylim(xx2.min(), xx2.max())
   colors = ["red", "yellow", "purple"]
   colormap=ListedColormap(colors)
#    plt.scatter(x=test_data.T[0], y=test_data.T[1],alpha=0.4, c=test_labels+3, cmap=colormap)

   # plot class samples
   for pp in (1, -1):
       labels = test_labels[np.where(test_labels==pp)[0]]
       labels = [x + 3 for x in labels]
       ccc= np.zeros(len(labels))+50.0
       plt.scatter(x=test_data.T[0][np.where(test_labels==pp)[0]], y=test_data.T[1][np.where(test_labels==pp)[0]],
       alpha=0.4, c = ccc, cmap=colormap)
       plt.show()
   

# decision region plot
plot_decision_regions(X, y, classifier=ada)

plt.title('Adaptive Linear Neuron - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

# plt.plot(range(1, len(aln.cost) + 1), aln.cost, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Sum-squared-error')
# plt.show()