import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.data import iris_data
from mlxtend.plotting import plot_decision_regions
from mlxtend.classifier import Adaline
# from partD import Adaline


# Initializing Classifiers
def plot_res(X, y, classifier):


    y= y.astype(int)
    # X[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    # X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    y= y.reshape(y.size,1).flatten()
    # Loading some example data
    # X = X[:,[0, 2]]

    # Plotting Decision Regions

    gs = gridspec.GridSpec(1, 1)
    fig = plt.figure(figsize=(3, 3))

    labels = ['Neural Network - two hidden layers']

    for clf, lab, grd in zip([classifier],
                            labels,
                            itertools.product([0, 1],
                            repeat=2)):
        # clf.fit( X, y)
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X=X, y=y,
                                    clf=clf, legend=2)
        plt.title(lab)

    plt.show()
