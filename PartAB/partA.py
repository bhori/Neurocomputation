import numpy as np
import random
# from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style

def save(features_name=None, labels_name=None, train=False, index=None, size=1000):
    data_x, data_y = create_data(size)
    if train==True:
        np.save("train", data_x)
        np.save("lables", data_y)
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
            coin2 = random.randint(0, 2)
            if coin2==0:
                data_x = np.append(data_x, int(random.uniform(-100, 50)) / 100)
                data_x = np.append(data_x, int(random.uniform(-100, 100)) / 100)
            if coin2==1:
                data_x = np.append(data_x, int(random.uniform(-100, 100)) / 100)
                data_x = np.append(data_x, int(random.uniform(-100, 50)) / 100)
            if coin2==2:
                data_x = np.append(data_x, int(random.uniform(-100, 50)) / 100)
                data_x = np.append(data_x, int(random.uniform(-100, 50)) / 100)
        else:
            data_x = np.append(data_x, int(random.uniform(50, 100)) / 100)
            data_x = np.append(data_x, int(random.uniform(50, 100)) / 100)
        # data_x = np.append(data_x, random.randint(-100, 100)/100)
        # data_x = np.append(data_x, random.randint(-100, 100)/100)
    for i in range(0,2*size,2):
        if data_x[i]>0.5 and data_x[i+1]>0.5:
            lables = np.append(lables, 1)
        else:
            lables = np.append(lables, -1)
    data_x = data_x.reshape(size, 2)
    lables = lables.reshape(size, 1)
    return (data_x.astype(float), lables.astype(float))

def check_validity(data_x, data_y):
    for i in range(100):
        if data_y[i][0]==1:
            print(data_x[i][0], ",", data_x[i][1],  "==>", data_y[i][0])

def train(data_x, data_y):
    w = [random.uniform(-1,1), random.uniform(-1,1)]
    b = random.uniform(-1,1)
    alpha = 0.0001
    for i in range(1000):
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
    print(w[0], w[1], b)
    sum = 0
    good=0
    prediction = np.array([])
    for i in range(1000):
        pred = w[0]*data_x[i][0]+w[1]*data_x[i][1]+b
        if pred>=0:
            prediction = np.append(prediction, 1)
        else:
            prediction = np.append(prediction, -1)
        if (pred >= 0 and data_y[i]==1) or (pred < 0 and data_y[i]==-1):
            good+=1
        correct_prediction = (pred - data_y[i])**2/1000
        sum += correct_prediction
    print(sum)
    print(good)
    print(good/1000)
    return (w[0], w[1], b, prediction)

def create_test(size=1000):
    save("test_data", "test_labels", size=size)
    # data_x, data_y = create_data(size)
    # np.save("test_data", data_x)
    # np.save("test_lables", data_y)

def create_large_train(size=1000):
    save("large_train", "large_labels", size=size)
    # data_x, data_y = create_data(size)
    # np.save("large_train", data_x)
    # np.save("large_lables", data_y)

def test_accuracy(w1, w2, b):
    # for i in range(1, 21):
        # print("file "+str(i)+":")
    data_x = np.load("test_data.npy")
    data_y = np.load("../test_labels.npy")
    # sum=0
    # for i in range(1000):
    #     if data_y[i][0]==1:
    #         sum+=1
    # #         print(data_x[i][0], ",", data_x[i][1],  "==>", data_y[i][0])
    # print("sum =", sum)

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
    print(sum)
    print(good)
    print("test accuracy: "+str(good / 1000))
    print("**************")

def show(data_x, data_y, prediction):

    # setting a custom style to use
    style.use('ggplot')

    # create a new figure for plotting
    fig = plt.figure()
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)

    X, Y = np.meshgrid(x, y)
    Z = -1 * X + -1 * Y + 1.5
    ax1 = fig.add_subplot(111, projection='3d')
    surf = ax1.plot_surface(X, Y, Z)

    # create a new subplot on our figure
    # and set projection as 3d

    # defining x, y, z co-ordinates

    x = data_x.T
    x = x[0]
    y = data_x.T
    y = y[0]
    z = prediction

    for i in x.size:
        print(x[i], y[i])

    # plotting the points on subplot
    ax1.scatter(x, y, z, c='m', marker='o')
    # setting labels for the axes
    ax1.set_xlabel('x-axis')
    ax1.set_ylabel('y-axis')
    ax1.set_zlabel('z-axis')

    # function to show the plot
    plt.show()

if __name__ == '__main__':
    # for i in range(1, 103):
    save(train=True)

    # for i in range(1, 11):
    #     data_x, data_y = load(i)
    #     print("*****")
    #     print(i)
    #     print("*****")
    #     check_validity(data_x, data_y)

    create_test()
    # create_large_train()

    # data_x = np.load("../train.npy")
    # data_y = np.load("../lables.npy")
    # w1, w2, b, prediction = train(data_x, data_y)
    # test_accuracy(w1, w2, b)


    # for i in range(100, 103):
    # print("file "+str(i)+":")
        # data_x, data_y = load(i)

    # sum = 0
    # for i in range(1000):
    #     if data_y[i][0]==1:
    #         sum+=1
    #         print(data_x[i][0], ",", data_x[i][1],  "==>", data_y[i][0])
    # print("sum =", sum)
        # w1, w2, b, prediction = train(data_x, data_y)
        # test_accuracy(w1, w2, b)
        # show(data_x, data_y, prediction)




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