import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob

data_x = np.load('../train.npy')
data_y = np.load('../lables.npy')
data_y = data_y.flatten()
test_data = np.load('../test_data.npy')
test_labels = np.load('../test_labels.npy')
test_labels = test_labels.flatten()
model = models.Sequential()
model.add(layers.Dense(2, activation='relu', input_shape=(2, 2)))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

history = model.fit(x=data_x, y=data_y, batch_size=None, epochs=400)

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