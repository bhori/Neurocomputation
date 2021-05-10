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


from ttt import AdalineGD

def input_new(data_x,weight_layer,bias_layer):
    big_array = []
    #for every record
    for i in range(data_x.T[0].size):
        array = []
        # for every neuron
        for i, (w, b) in enumerate(zip(weight_layer,bias_layer)):
            pred = w[0] * data_x[i][0] + w[1] * data_x[i][1] + b
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

data_x = np.load('train.npy')
labels = np.load('lables.npy')
labels = labels.flatten()

# print_data(data_x, labels)
# labels = labels.flatten()
test_data = np.load('test_data.npy')
test_labels = np.load('test_labels.npy')
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
x = tf.keras.layers.Dense(4, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(1, activation='linear')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


model.summary()

model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

history = model.fit(x=data_x, y=labels, epochs=10)

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
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(i), marker=markers[4], label=cl)
plot_decision_regions(data_x, labels, classifier=model)

plt.title(' 2 Layers - nonlinear learning')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

inputs_n = tf.keras.Input(shape=(2,))
outputs_n = tf.keras.layers.Dense(1, activation='sigmoid')(inputs_n)
model_n = tf.keras.Model(inputs=inputs_n, outputs=outputs_n)
W_and_b =model.layers[1].get_weights()
print(W_and_b)
weight_layer=W_and_b[0].T
bias_layer= W_and_b[1]
input_new(data_x,weight_layer,bias_layer)
adasgd = AdalineGD(n_iter=200, eta=0.01)
adasgd.fit(data_x, labels)



# for i, (w, b) in enumerate(zip(weight_layer,bias_layer)):
#     l=[]
#     w= np.array(w)
#     num_of_w = w.size
#     w=np.reshape(w,(num_of_w,1))
#     b= np.array(b)
#     b=np.reshape(b,(1,))
#     l.append(w)
#     l.append(b)
#     l = l.flatten()
#     # model_n.layers[1].set_weights(l)
#     # print(model_n.layers[1].get_weights())
#     # plot_decision_regions(data_x, labels, classifier=model_n)
#     # plt.title('the learning of layer 1 - neuron '+str(i))
#     # plt.xlabel('sepal length [standardized]')
#     # plt.ylabel('petal length [standardized]')
#     # plt.legend(loc='upper left')
#     # plt.show()
#     ada = AdalineGD(n_iter=200, eta=0.01)
#     ada.fit(data_x,labels)
plot_decision_regions(data_x, labels, classifier=adasgd)
plt.title('the learning of layer 1 - neuron ')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()