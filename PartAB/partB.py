import numpy as np
import random
import math 
# from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style

def save(features_name=None, labels_name=None, train=False, index=None, size=1000):
    data_x, data_y = create_data(size)
    if train==True:
        np.save("train_b", data_x)
        np.save("lables_b", data_y)
    else:
        np.save(features_name, data_x)
        np.save(labels_name, data_y)

def load(index):
    return (np.load("../train.npy"), np.load("../lables.npy"))
    # return (np.load("train"+str(index)+".npy"), np.load("lables"+str(index)+".npy"))

def create_data(size=1000):
    data_x = np.array([])
    lables = np.array([])
    for i in range(size):
        # data_x = np.random.uniform(-1.0, 1.0, 2000)
        coin = random.randint(0, 1)
        # print(coin)
        if coin==0:
            coin2 = random.uniform(0.5, 0.75)
            sqrt_coin2= math.sqrt(coin2)
            coin_x= random.uniform(-1*sqrt_coin2,sqrt_coin2)
        
            coin_m=  random.randint(0, 1)
            if coin_m==0 :
                y= -1*math.sqrt(coin2-coin_x**2)
            else :
                y= math.sqrt(coin2-coin_x**2)
            data_x = np.append(data_x,  int(coin_x*100)/100 )
            data_x = np.append(data_x, int(y*100)/100 )
        else:
            coin_z = random.randint(0, 2)
            if coin_z==2:
                coin2= random.uniform(0.75, 1.0)
            else:
                coin2= random.uniform(0, 0.5)
            sqrt_coin2= math.sqrt(coin2)
            coin_x= random.uniform(-1*sqrt_coin2,sqrt_coin2)
            coin_m=  random.randint(0, 1)
            if coin_m==0 :
                y= -1*math.sqrt(coin2-coin_x**2)
            else :
                y= math.sqrt(coin2-coin_x**2)
            data_x = np.append(data_x,  int(coin_x*100)/100 )
            data_x = np.append(data_x, int(y*100)/100 )
        # data_x = np.append(data_x, random.randint(-100, 100)/100)
        # data_x = np.append(data_x, random.randint(-100, 100)/100)
    count1=0

    for i in range(0,2*size,2):
        if data_x[i]**2+data_x[i+1]**2>0.5 and  0.75>data_x[i]**2+data_x[i+1]**2:
            lables = np.append(lables, 1)
            count1=count1+1
        else:
            lables = np.append(lables, 0)
        # print(data_x[i]**2+data_x[i+1]**2)
    print(count1)
    data_x = data_x.reshape(size, 2)
    lables = lables.reshape(size, 1)
    return (data_x.astype(float), lables.astype(float))

    # for i in range(1000):
    #     if lables[i][0]==1:
    #      print(data_x[i][0], ",", data_x[i][1],  "==>", lables[i][0], data_x[i][1]**2+data_x[i][0]**2 )


def check_validity(data_x, data_y):
    for i in range(1000):
        if data_y[i][0]==1:
            print(data_x[i][0], ",", data_x[i][1],  "==>", data_y[i][0])

def train(data_x, data_y):
    w = [0, 0]
    b = 0
    alpha = 0.01
    for i in range(500):
        # print("Epoch ", i, ":", w[0], w[1], b)
        for j,obj in enumerate(data_x):
            x1,x2 = obj[0], obj[1]
            middle_ans = w[0]*x1+w[1]*x2+b
            if middle_ans>=0:
                y_in = 1
            else:
                y_in = -1
            b = b + alpha*(data_y[j]-middle_ans)
            w[0] = w[0] + alpha*(data_y[j]-middle_ans)*x1
            w[1] = w[1] + alpha*(data_y[j] - middle_ans)*x2
    # print(w[0], w[1], b)
    sum = 0
    good=0
    prediction = np.array([])
    for i in range(1000):
        pred = w[0]*data_x[i][0]+w[1]*data_x[i][1]+b
        if pred>=0:
            print(data_x[i][1]**2+data_x[i][0]**2)
            prediction = np.append(prediction, 1)
        else:
            prediction = np.append(prediction, -1)
        if (pred >= 0 and data_y[i]==1) or (pred < 0 and data_y[i]==-1):
            good+=1
        correct_prediction = (pred - data_y[i])**2/1000
        sum += correct_prediction
    return (w[0], w[1], b, prediction)

def create_test():
    data_x, data_y = create_data()
    np.save("test_data_b", data_x)
    np.save("test_lables_b", data_y)

def create_large_train():
    data_x, data_y = create_data()
    np.save("large_train", data_x)
    np.save("large_lables", data_y)

def test_accuracy(w1, w2, b):
    # for i in range(1, 21):
        # print("file "+str(i)+":")
    data_x = np.load("test_data_b.npy")
    data_y = np.load("test_lables_b.npy")
    # for i in range(1000):
    #     if data_y[i][0]==1:
    #         print(data_x[i][0], ",", data_x[i][1],  "==>", data_y[i][0])

    sum = 0
    good = 0
    for i in range(1000):
        pred = w1 * data_x[i][0] + w2 * data_x[i][1] + b
        # pred = data_x[i][0] + data_x[i][1] - 1.5
        if (pred >= 0 and data_y[i] == 1) or (pred < 0 and data_y[i] == -1):
            good += 1
        correct_prediction = (pred - data_y[i]) ** 2 / 1000
        sum += correct_prediction
    # print("sum= ",sum)
    # print(good)
    # print("test accuracy: "+str(good / 1000))
    # print("**************")

def show(data_x, data_y, prediction, W1, W2, b):

    # setting a custom style to use
    style.use('ggplot')

    # create a new figure for plotting
    fig = plt.figure()
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)

    X, Y = np.meshgrid(x, y)
    Z = W1 * X + W2 * Y + b
    ax1 = fig.add_subplot(111, projection='3d')
    surf = ax1.plot_surface(X, Y, Z)

    # create a new subplot on our figure
    # and set projection as 3d

    # defining x, y, z co-ordinates

    x = data_x.T
    x = x[0]
    y = data_x.T
    y = y[1]
    z = prediction

    
    # print(x, y)

    # plotting the points on subplot
    ax1.scatter(x, y, z, c='m', marker='o')
    # setting labels for the axes
    ax1.set_xlabel('x-axis')
    ax1.set_ylabel('y-axis')
    ax1.set_zlabel('z-axis')

    # function to show the plot
    plt.show()

if __name__ == '__main__':
    # for i in range(21, 31):
        save(train=True)
        create_test()
    # for i in range(1, 11):
    #     data_x, data_y = load(i)
    #     print("*****")
    #     print(i)
    #     print("*****")
    #     check_validity(data_x, data_y)

    
    # create_large_train()

    # data_x = np.load("large_train.npy")
    # data_y = np.load("large_lables.npy")
    # w1, w2, b = train(data_x, data_y)
    # test_accuracy(w1, w2, b)

    # print("file "+str(i)+":")
    # data_x, data_y = load(21)
    # w1, w2, b, prediction = train(data_x, data_y)
    # for i in range(22, 31):
    #        # create_test()
    #     data_x, data_y=load(i)
    #     test_accuracy(w1, w2, b)
    #     show(data_x, data_y, prediction,w1, w2, b)




# def data_x():
#     file = open("dataA.txt", "r")
#     data_x = np.array([])
#     count = 0
#     for i,line in enumerate(file):
#         count = count + 1
#         data = line.split()
#         x = np.array(data)
#         data_x = np.append(data_x, x)
#     data_x = data_x.reshape(count, 2)
#     return data_x.astype(float)
#
# def data_y():
#     file = open("resultA.txt", "r")
#     data_y = np.array([])
#     count = 0
#     for i,line in enumerate(file):
#         count = count + 1
#         y = np.array(line)
#         data_y = np.append(data_y, y)
#     data_y = data_y.reshape(count, 1)
#     return data_y.astype(float)