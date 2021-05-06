from numpy.random import seed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        SSE in every epoch.

    """
    def __init__(self,eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self, X, y):
        """ Fit training data. 

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(X.shape[1] + 1)
        self.cost_ = []
        
        for i in range(self.n_iter):
            # calculating wTx
            output = self.net_input(X)

            # calculating errors
            errors = y - output
            
            # calculating weight update
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            
            # calculating cost
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        return self.net_input(X)
        # Please note that the "activation" method has no effect
        # in the code since it is simply an identity function. We
        # could write `output = self.net_input(X)` directly instead.
        # The purpose of the activation is more conceptual, i.e.,  
        # in the case of logistic regression, we could change it to
        # a sigmoid function to implement a logistic regression classifier.
        
    def predict(self, X):
        return np.where(self.activation(X) >= 0.5, 1, 0)



def plot_decision_regions(X, y, classifier, resolution=0.02):
# setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('r', 'b', 'g', 'k', 'grey')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision regions by creating a pair of grid arrays xx1 and xx2 via meshgrid function in Numpy
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
    # use predict method to predict the class labels z of the grid points
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    # draw the contour using matplotlib
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class samples
    for i, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(i), marker=markers[i], label=cl)
# X = np.load("train10.npy")
# y =  np.load("lables10.npy")
# y= y.flatten()
# df = pd.read_csv('https://archive.ics.uci.edu/ml/'
#         'machine-learning-databases/iris/iris.data', header=None)

# # select two classes: setosa and versicolor
# y = df.iloc[0:100, 4].values  # values method of a pandas dataframe yields Numpy array
# y = np.where(y == 'Iris-setosa', -1, 1)

# # select two features: sepal length and petal length for visualization
# X = df.iloc[0:100, [0,2]].values

# # standardize features
# X_std = np.copy(X)
# X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
# X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
# adasgd = AdalineGD(n_iter=200, eta=0.01)
# adasgd.fit(X, y)

# plot_decision_regions(X, y, classifier=adasgd)

# plt.title('Adaline - Stochastic Gradient Descent')
# plt.xlabel('sepal length [standardized]')
# plt.ylabel('petal length [standardized]')
# plt.legend(loc='upper left')
# plt.show()

# plt.plot(range(1, len(adasgd.cost_) + 1), adasgd.cost_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Average Cost')
# plt.show()