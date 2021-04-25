import numpy as np
import random

def create_data():
    # Random rand = new Random();
    # x = []
    y = []
    data_x = np.array([])
    lables = np.array([])
    for i in range(1000):
        data_x = np.append(data_x, random.randint(-100, 100)/100)
        data_x = np.append(data_x, random.randint(-100, 100)/100)
        # data_x.append(random.randint(-100, 100))
        # data_x.append(random.randint(-100, 100))
    for i in range(0,2000,2):
        if data_x[i]>0.5 and data_x[i+1]>0.5:
            lables = np.append(lables, 1)
        else:
            lables = np.append(lables, -1)
        # i=i+1
    data_x = data_x.reshape(1000, 2)
    lables = lables.reshape(1000, 1)
    return (data_x.astype(float), lables.astype(float))


def data_x():
    file = open("dataB.txt", "r")
    data_x = np.array([])
    count = 0
    for i,line in enumerate(file):
        count = count + 1
        data = line.split()
        x = np.array(data)
        data_x = np.append(data_x, x)
    data_x = data_x.reshape(count, 2)
    return data_x.astype(float)

def data_y():
    file = open("resultB.txt", "r")
    data_y = np.array([])
    count = 0
    for i,line in enumerate(file):
        count = count + 1
        # data = line.split()
        y = np.array(line)
        data_y = np.append(data_y, y)
    data_y = data_y.reshape(count, 1)
    return data_y.astype(float)

if __name__ == '__main__':
    # data_x = data_x()
    # data_y = data_y()

    data_x, data_y = create_data()

    # for i in range(1000):
    #     if data_y[i][0]==1:
    #         print(data_x[i][0], ",", data_x[i][1],  "==>", data_y[i][0])
    # print(data_y)

    # data_x, data_y = create_data()

    w = [0, 0]
    b = 0
    alpha = 0.0001
    for i in range(500):
        # print(w[0], w[1], b)
        for (j,obj) in enumerate(data_x):
            x1,x2 = obj[0], obj[1]
            middle_ans = w[0]*x1+w[1]*x2+b
            if middle_ans>=0:
                y_in = 1
            else:
                y_in = -1
            # print(data_y[j]-y_in)
            b = b + alpha*(data_y[j]-y_in)
            w[0] = w[0] + alpha*(data_y[j]-y_in)*x1
            w[1] = w[1] + alpha*(data_y[j] - y_in)*x2
    print(w[0], w[1], b)
    sum = 0
    good=0
    for i in range(1000):
        pred = w[0]*data_x[i][0]+w[1]*data_x[i][1]+b
        if (pred >= 0 and data_y[i]==1) or (pred < 0 and data_y[i]==-1):
            good+=1
        correct_prediction = (pred - data_y[i])**2/1000
        sum += correct_prediction
    print(sum)
    print(good)
    print(good/1000)

    # for i in range(0,10000):
    #     # print("i=", i, "==>", 2**i)
    #     sum = 0
    #     for j in range(0, i):
    #         sum+=2**j
    #     print("i=", i, "==>", (2 ** i) - sum)