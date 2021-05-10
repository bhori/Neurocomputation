import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob

# class Net():
#     def __init__(self, number_of_neurons=2):
#         inputs = tf.keras.Input(shape=(2,))
#         x = tf.keras.layers.Dense(number_of_neurons, activation='relu')(inputs)
#         outputs = tf.keras.layers.Dense(1, activation='tanh')(x)
#         self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
#         self.history = None
#
#     def summary(self):
#         self.model.summary()
#
#     def train(self, data, labels, optimizer='adam',loss=keras.losses.MeanSquaredError(), epochs = 400):
#         self.model.compile(optimizer=optimizer,
#                       loss=loss,  # keras.losses.MeanSquaredError 'binary_crossentropy'
#                       metrics=['accuracy'])
#
#         self.history = model.fit(x=data, y=labels, epochs=epochs)
#
#     def show(self, test_data, test_labels):
#         # Show results in graph view
#         plt.clf()
#         plt.plot(self.history.history['accuracy'], label='accuracy')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')
#         plt.ylim([0, 1])
#         plt.legend(loc='lower right')
#         # plt.show()
#
#         test_loss, test_acc = self.model.evaluate(test_data, test_labels, verbose=2)
#
#         plt.show()
#
#     def save(self, number_of_neurons):
#         # Show results in graph view
#         plt.clf()
#         plt.title(str(number_of_neurons) + ' neurons')
#         plt.plot(self.history.history['accuracy'], label='accuracy')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')
#         plt.ylim([0, 1])
#         plt.legend(loc='lower right')
#
#         test_loss, test_acc = self.model.evaluate(test_data, test_labels, verbose=2)
#
#         plt.savefig('Images/partC/' + str(number_of_neurons) + '.png')


def build(number_of_neurons=2):
    inputs = tf.keras.Input(shape=(2,))
    x = tf.keras.layers.Dense(number_of_neurons, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(1, activation='tanh')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def train(model, data, labels, optimizer='adam',loss=keras.losses.MeanSquaredError(), epochs = 400):
    model.compile(optimizer=optimizer,
                  loss=loss,  # keras.losses.MeanSquaredError 'binary_crossentropy'
                  metrics=['accuracy'])

    history = model.fit(x=data, y=labels, epochs=epochs)
    return history

def load(file_name):
    return np.load(file_name)

def print_data(data, labels):
    for i in range(1000):
        if labels[i][0]==1:
            print(data[i][0], ",", data[i][1],  "==>", labels[i][0])

def show(model, history, test_data, test_labels):
    # Show results in graph view
    plt.clf()
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    # plt.show()

    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)

    plt.show()

def save(model, history, number_of_neurons):
    # Show results in graph view
    plt.clf()
    plt.title(str(number_of_neurons)+' neurons')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)

    plt.savefig('Images/partC/'+str(number_of_neurons)+'.png')


if __name__ == '__main__':
    data_x = load('../train.npy')
    labels = load('../lables.npy')
    test_data = load('../test_data.npy')
    test_labels = load('../test_labels.npy')
    for i in range(2, 10):
        model = build(number_of_neurons=i)
        model.summary()
        history = train(model, data_x, labels)
        show(model, history, test_data, test_labels)
        # save(model, history, number_of_neurons=i)

    # for i in range(2, 10):
    #     model = Net(number_of_neurons=i)
    #     model.summary()
    #     model.train(model, data_x, labels)
    #     model.show(model, model.history, test_data, test_labels)
    #     # save(model, history, number_of_neurons=i)

# The good version before separating to functions:

# inputs = tf.keras.Input(shape=(2,))
# x = tf.keras.layers.Dense(100, activation='relu')(inputs)
# outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)


# model.compile(optimizer='adam',
#                   loss= keras.losses.MeanSquaredError(), # keras.losses.MeanSquaredError 'binary_crossentropy'
#                   metrics=['accuracy'])
#
# history = model.fit(x=data_x, y=labels, epochs=400)

# # Show results in graph view
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0, 1])
# plt.legend(loc='lower right')
# # plt.show()
#
# test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
#
# plt.show()




# Other versions:

# model = keras.models.Sequential()
#
# original_inputs = tf.keras.Input(shape=(2,), name="encoder_input")
# x = layers.Dense(2, activation="relu")(original_inputs)
# model = keras.Model()

# model = keras.models.Sequential()
# model.add(keras.Input(shape=(2,)))
# model.add(layers.Dense(100, activation='relu')) # , input_shape=(2, )
# model.add(layers.Dense(1))
