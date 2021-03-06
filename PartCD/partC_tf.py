import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
from matplotlib.colors import ListedColormap
import os
import sys
from mlxtend.classifier import Adaline
import matplotlib.gridspec as gridspec

from plot import plot_res
from ttt import AdalineGD


from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools
# def plot_result(X,y,classifier):
#     grd = list(itertools.product([0, 1]))
#     xx , yy = np.meshgrid(np.linspace(-3, 3, 50),np.linspace(-3, 3, 50))
#     gs = gridspec.GridSpec(2, 2)
#     fig = plt.figure(figsize=(10,8))
#     ax = plt.subplot(gs[grd[0], grd[1]])
#     fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
#     plt.title(lab)
#     plt.show()


def input_new(data_x,weight_layer,bias_layer):
    big_array = []
    #for every record
    for i in range(len(data_x)):
        array = []
        # for every neuron
        for j , (w, b) in enumerate(zip(weight_layer,bias_layer)):
            pred = b
            for k in range(len(w)):
                pred += w[k] * data_x[i][k]
            # pred += b
            #relu
            if pred<0 :
                pred=0
            array.append(pred)
        array=np.array(array)
        big_array.append(array)
    big_array=np.array(big_array)
    print(big_array.shape)
    return big_array


def print_data(data, labels):
    for i in range(1000):
        if labels[i][0]==1:
            print(data[i][0], ",", data[i][1],  "==>", labels[i][0])

data_x = np.load('../train_b.npy')
labels = np.load('../lables_b.npy')
labels = labels.flatten()

# print_data(data_x, labels)
# labels = labels.flatten()
test_data = np.load('../test_data_b.npy')
test_labels = np.load('../test_lables_b.npy')
# print_data(test_data, test_labels)
# test_labels = test_labels.flatten()

# model = keras.models.Sequential()
#
# original_inputs = tf.keras.Input(shape=(2,), name="encoder_input")
# x = layers.Dense(2, activation="relu")(original_inputs)
# model = keras.Model()

# model = keras.models.Sequential()
# model.add(keras.Input(shape=(2,)))
# model.add(layers.Dense(100, activation='relu')) # , input_shape=(2, )
# model.add(layers.Dense(1))

inputs = tf.keras.Input(shape=(2,))
c= tf.keras.layers.Dense(7, activation='relu')(inputs)
x = tf.keras.layers.Dense(4, activation='relu')(c)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


model.summary()

model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

history = model.fit(x=data_x, y=labels, epochs=400, batch_size=20)

# Show results in graph view
plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
# plt.show()

test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)

plt.show()


# def plot_decision_regions(X, y, classifier, resolution=0.1):
# # setup marker generator and color map
#     markers = ('s', 'x', 'o', '^', 'v')
#     colors = ('r', 'b', 'g', 'k', 'grey')
#     cmap = ListedColormap(colors[:len(np.unique(y))])
    
#     # plot the decision regions by creating a pair of grid arrays xx1 and xx2 via meshgrid function in Numpy
#     x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
#     hh= np.array([xx1.ravel(),xx2.ravel()]).T
#     print(hh.shape)
#     # use predict method to predict the class labels z of the grid points
#     Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
#     Z = Z.reshape(xx1.shape)
    
#     # draw the contour using matplotlib
#     plt.contourf(xx1, xx2, Z, alpha=0.7, cmap=cmap)
#     plt.xlim(xx1.min(), xx1.max())
#     plt.ylim(xx2.min(), xx2.max())
    
#     # plot class samples
#     for i, cl in enumerate(np.unique(y)):
#         plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.5, c=cmap(i), marker=markers[4], label=cl)
plot_res(data_x, labels, classifier=model)

# plt.title(' 2 Layers - nonlinear learning')
# plt.xlabel('sepal length [standardized]')
# plt.ylabel('petal length [standardized]')
# plt.legend(loc='upper left')
# plt.show()

inputs_n = tf.keras.Input(shape=(2,))
outputs_n = tf.keras.layers.Dense(1, activation='relu')(inputs_n)
model_n = tf.keras.Model(inputs=inputs_n, outputs=outputs_n)
W_and_b =model.layers[1].get_weights()
print(W_and_b)
weight_layer=W_and_b[0].T
bias_layer= W_and_b[1]
inputn=input_new(data_x,weight_layer,bias_layer)


X=data_x
# X[:,0] = (data_x[:,0] - data_x[:,0].mean()) / data_x[:,0].std()
# X[:,1] = (data_x[:,1] - data_x[:,1].mean()) / data_x[:,1].std()

y= labels.astype(np.integer)

# gs = gridspec.GridSpec(2, 1)
# fig = plt.figure(figsize=(10, 8))
#part c
# for i, (w, b)  in enumerate(zip(weight_layer,bias_layer)):
#     # l=[]
#     # w= np.array(w)
#     # num_of_w = w.size
#     # w=np.reshape(w,(num_of_w,1))
#     # b= np.array(b)
#     # b=np.reshape(b,(1,))
#     # l.append(w)
#     # l.append(b)
#     ada = Adaline(epochs=15,
#     eta=0.02,
#     minibatches=1000, # for SGD learning w. minibatch size 20
#     random_seed=1,
#     print_progress=3)
#     y= labels.astype(np.integer)
#     ada.fit(X, y)
#     ada.w_= w
#     ada.b_ =b
#     # plt.show()
#     # fig=plot_decision_regions(X, y, clf=ada)
#     plot_decision_regions(X, y, clf=ada)
#     plt.title('First hidden layer neuron '+str(i))
#     plt.show()
#     # plt.plot(range(len(ada.cost_)), ada.cost_)
#     # plt.xlabel('Iterations')
#     # plt.ylabel('Cost')
#     # plt.show()
#     # model_n.layers[1].set_weights(l)
#     # print(model_n.layers[1].get_weights())
#     # model_n.summary()
#     # model_n.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])
#     # print("after compile")
#
#     # print(model_n.layers[1].get_weights())
#     # plot_res(data_x, labels, classifier=model_n)
#
#     #---------------------------------------------
#     # plt.title('the learning of layer 1 - neuron '+str(i))
#     # plt.xlabel('sepal length [standardized]')
#     # plt.ylabel('petal length [standardized]')
#     # plt.legend(loc='upper left')
#     # plt.show()

inputs_n = tf.keras.Input(shape=(2,))
middle = tf.keras.layers.Dense(7, activation='relu')(inputs_n)
outputs_n = tf.keras.layers.Dense(1, activation='relu')(middle)
model_n = tf.keras.Model(inputs=inputs_n, outputs=outputs_n)
W_and_b_2 =model.layers[2].get_weights()
print(W_and_b_2)
weight_layer_2=W_and_b_2[0].T
bias_layer_2= W_and_b_2[1]
inputn_2=input_new(inputn,weight_layer_2,bias_layer_2)

for i, (w, b) in enumerate(zip(weight_layer_2,bias_layer_2)):
    l=[]
    w= np.array(w)
    num_of_w = w.size
    w=np.reshape(w,(num_of_w,1))
    b= np.array(b)
    b=np.reshape(b,(1,))
    l.append(w)
    l.append(b)
    plt.title('Second hidden layer neuron '+str(i))
    # plt.show()
    # plt.plot(range(len(ada.cost_)), ada.cost_)
    # plt.xlabel('Iterations')
    # plt.ylabel('Cost')
    # plt.show()
    model_n.layers[1].set_weights(model_n.layers[1].get_weights())
    model_n.layers[2].set_weights(l)
    print(model_n.layers[2].get_weights())
    model_n.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])
    print("after compile")

    print(model_n.layers[2].get_weights())
    plot_decision_regions(data_x, y, clf=model_n)
    plt.title('Second hidden layer neuron ' + str(i))
    plt.show()
    # plot_res(data_x, labels, classifier=model_n)

#part d
#this is try with other adaline

ada = Adaline(epochs=15,eta=0.02,minibatches=1000, # for SGD learning w. minibatch size 20
random_seed=1,
print_progress=3)
# y= labels.astype(np.integer)
ada.fit(inputn_2, y)
print(ada.score(inputn_2, y))
l = []
w = np.array(ada.w_)
num_of_w = w.size
w = np.reshape(w, (num_of_w, 1))
b = np.array(ada.b_)
b = np.reshape(b, (1,))
l.append(w)
l.append(b)
model.layers[3].set_weights(l)

plot_decision_regions(data_x, y, clf=model_n)
plt.title('Neurons from the next to last level using Adaline')
plt.show()


# adasgd = AdalineGD(n_iter=200, eta=0.01)
# adasgd.fit(inputn_2, labels)



# model_n.summary()
# model_n.compile(optimizer='adam',
#                   loss=tf.keras.losses.MeanSquaredError(),
#                   metrics=['accuracy'])

# labels= labels.reshape(labels.size,1)
# history = model_n.fit(x=inputn, y=labels, epochs=10)
# plot_res(inputn, labels, classifier=model_n)
# plt.title('the learning of layer 1 - neuron ')
# plt.xlabel('sepal length [standardized]')
# plt.ylabel('petal length [standardized]')
# plt.legend(loc='upper left')
# plt.show()
