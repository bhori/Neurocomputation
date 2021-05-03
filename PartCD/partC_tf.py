import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob


def print_data(data, labels):
    for i in range(1000):
        if labels[i][0]==1:
            print(data[i][0], ",", data[i][1],  "==>", labels[i][0])

data_x = np.load('../train.npy')
labels = np.load('../lables.npy')
# print_data(data_x, labels)
# labels = labels.flatten()

test_data = np.load('../test_data.npy')
test_labels = np.load('../test_labels.npy')
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
x = tf.keras.layers.Dense(100, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


model.summary()

model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

history = model.fit(x=data_x, y=labels, epochs=400)

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