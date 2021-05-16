import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.classifier import Adaline
from mlxtend.plotting import plot_decision_regions


#create data of next layer input by applay current layer weight and bias on this layer input
def input_new(X,weight_layer,bias_layer):
    big_array = []
    #for every record
    for i in range(len(X)):
        array = []
        # for every neuron
        for j , (w, b) in enumerate(zip(weight_layer,bias_layer)):
            pred = b
            for k in range(len(w)):
                pred += w[k] * X[i][k]
            #relu
            if pred<0 :
                pred=0
            array.append(pred)
        array=np.array(array)
        big_array.append(array)
    big_array=np.array(big_array)
    # print(big_array.shape)
    return big_array

#print data and labels
def print_data(data, labels):
    for i in range(1000):
        if labels[i][0]==1:
            print(data[i][0], ",", data[i][1],  "==>", labels[i][0])

#load data
X = np.load('train_b.npy')
labels = np.load('lables_b.npy')
test_data = np.load('test_data_b.npy')
test_labels = np.load('test_lables_b.npy')
labels= labels.astype(np.integer)
labels= labels.flatten()





#Model architecture
inputs = tf.keras.Input(shape=(2,))
c= tf.keras.layers.Dense(7, activation='relu')(inputs)
x = tf.keras.layers.Dense(4, activation='relu')(c)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
history = model.fit(x=X, y=labels, epochs=1000, batch_size=97)

#save dada,prediction, class prediction, true label for all data record
dt= np.array(list(zip(X.T[0] ,X.T[1] ,model.predict(X),np.where(model.predict(X)>= 0.5, 1, 0),labels)))
np.savetxt("data-predict_class-predict_true-label.csv", dt , fmt="%f" ,delimiter=",")

#save learning weights for every neuron in each layer
np.savetxt("data-weights-layer1.csv", (model.layers[1].get_weights()[0][0],model.layers[1].get_weights()[0][1]
    ,model.layers[1].get_weights()[1]) ,fmt="%f",delimiter=",", header= "each column represent neuron and each row represent a weight when last row represent the bias")
np.savetxt("data-weights-layer2.csv", (model.layers[2].get_weights()[0][0],model.layers[2].get_weights()[0][1]
    ,model.layers[2].get_weights()[0][2],model.layers[2].get_weights()[0][3],model.layers[2].get_weights()[0][4],
    model.layers[2].get_weights()[0][5],model.layers[2].get_weights()[0][6],model.layers[2].get_weights()[1]) ,fmt="%f", delimiter=",", header= "each column represent neuron and each row represent a weight when last row represent the bias")
np.savetxt("data-weights-layer3.csv", (model.layers[1].get_weights()[0][0]
,model.layers[1].get_weights()[1]) ,fmt="%f",delimiter=",", header= "each column represent neuron and each row represent a weight when last row represent the bias")


# Show results in graph view
plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
plt.show()

fig = plt.figure(figsize=(3, 3))
#print diagram of the model learning on the dataset
fig=plot_decision_regions(X, labels, clf=model)
plt.show()






#create model to show result of each neuron in first hidden layer
inputs_n = tf.keras.Input(shape=(2,))
outputs_n = tf.keras.layers.Dense(1, activation='relu')(inputs_n)
model_n = tf.keras.Model(inputs=inputs_n, outputs=outputs_n)
W_and_b =model.layers[1].get_weights()
# print(W_and_b)
weight_layer=W_and_b[0].T
bias_layer= W_and_b[1]

#create data after applay second layer (the input data to output layer)
inputn=input_new(X,weight_layer,bias_layer)


#part c
for i, (w, b)  in enumerate(zip(weight_layer,bias_layer)):
    l=[]
    w= np.array(w)
    num_of_w = w.size
    w=np.reshape(w,(num_of_w,1))
    b= np.array(b)
    b=np.reshape(b,(1,))
    l.append(w)
    l.append(b)
    model_n.summary()
    model_n.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])
    model_n.layers[1].set_weights(l)
    fig = plt.figure(figsize=(3, 3))
#print diagram of the model learning on the dataset
    fig=plot_decision_regions(X, labels, clf=model_n)
    plt.title('the learning of layer 1 - neuron '+str(i))
    plt.show()


#create model to show result of each neuron in second hidden layer
inputs_n = tf.keras.Input(shape=(2,))
middle = tf.keras.layers.Dense(7, activation='relu')(inputs_n)
outputs_n = tf.keras.layers.Dense(1, activation='relu')(middle)
model_n2 = tf.keras.Model(inputs=inputs_n, outputs=outputs_n)
W_and_b_2 =model.layers[2].get_weights()
weight_layer_2=W_and_b_2[0].T
bias_layer_2= W_and_b_2[1]

#create data after applay second layer (the input data to output layer)
inputn_2=input_new(inputn,weight_layer_2,bias_layer_2)

#set the model weight and bias of output layer(for each neuron on secon layer)
for i, (w, b) in enumerate(zip(weight_layer_2,bias_layer_2)):
    l=[]
    w= np.array(w)
    num_of_w = w.size
    w=np.reshape(w,(num_of_w,1))
    b= np.array(b)
    b=np.reshape(b,(1,))
    l.append(w)
    l.append(b)
    model_n2.layers[1].set_weights(model.layers[1].get_weights())
    model_n2.layers[2].set_weights(l)
    model_n2.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])
    fig = plt.figure(figsize=(3, 3))
    #print diagram of the model learning on the dataset
    fig=plot_decision_regions(X, labels, clf=model_n2)
    plt.title('Second hidden layer neuron '+str(i))
    plt.show()


#part d

ada = Adaline(epochs=100,eta=0.05,minibatches=500, 
random_seed=1,
print_progress=3)
# y= labels.astype(np.integer)
ada.fit(inputn_2, labels)
acc= ada.score(inputn_2, labels)
print("\n prediction accuracy train: ",acc)

l = []
w = np.array(ada.w_)
num_of_w = w.size
w = np.reshape(w, (num_of_w, 1))
b = np.array(ada.b_)
b = np.reshape(b, (1,))
l.append(w)
l.append(b)
model.layers[3].set_weights(l)

fig = plt.figure(figsize=(3, 3))
#print diagram of the model learning on the dataset
fig=plot_decision_regions(X, labels, clf=model)
plt.title('Neurons from the next to last level using Adaline')
plt.show()

dt_ada= np.array(list(zip(X.T[0] ,X.T[1] ,ada.predict(inputn_2),labels)))
np.savetxt("data-predict_class-true-label_last_layer_adaline.csv", dt_ada , fmt="%f" ,delimiter=",")