import numpy as np
import random
# from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style


class Adaline(object):
    def __init__(self, w=None,b=None):
           # instance variable unique to each instance
        self.w=w
        self.b=b
    def fit(self, data_x, data_y):
        self.w = np.zeros(data_x.shape[1])
        self.b =0
        alpha = 0.01
        for i in range(100):
            # print("Epoch ", i, ":", w[0], w[1], b)
            for j,obj in enumerate(data_x):
                x1,x2 = obj[0], obj[1]
                middle_ans = self.w[0]*x1+self.w[1]*x2+self.b
                if middle_ans>=0:
                    y_in = 1
                else:
                    y_in = -1
                self.b = self.b + alpha*(data_y[j]-middle_ans)
                self.w[0] = self.w[0] + alpha*(data_y[j]-middle_ans)*x1
                self.w[1] = self.w[1] + alpha*(data_y[j] - middle_ans)*x2
        print(self.w[0], self.w[1], self.b)
        return self
    def predict(self, X):
        sum = 0
        good=0
        prediction = np.array([])
        for i in range(len(X)-1):
            pred = self.w[0]*X[i][0]+self.w[1]*X[i][1]+self.b
            if pred>=0.5:
                prediction = np.append(prediction, 1)
            else:
                prediction = np.append(prediction, 0)
            # if (pred >= 0 and data_y[i]==1) or (pred < 0 and data_y[i]==-1):
            #     good+=1
            # correct_prediction = (pred - data_y[i])**2/1000
            # sum += correct_prediction
        return prediction
        # self.test_accuracy(self.w[0], self.w[1], self.b)





    def test_accuracy(self,w1, w2, b):
        data_x = np.load("test_data.npy")
        data_y = np.load("test_labels.npy")
        sum = 0
        good = 0
        for i in range(1000):
            pred = w1 * data_x[i][0] + w2 * data_x[i][1] + b
            if (pred >= 0.5 and data_y[i] == 1) or (pred < 0.5 and data_y[i] == 0):
                good += 1
            correct_prediction = (pred - data_y[i]) ** 2 / 1000
            sum += correct_prediction
        print("sum= ",sum)
        print(good)
        print("test accuracy: "+str(good / 1000))
        print("**************")
# show(data_x, data_y, prediction,w1, w2, b)

