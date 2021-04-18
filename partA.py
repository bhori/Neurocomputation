import numpy as np

def data_x():
    file = open("dataA.txt", "r")
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
    file = open("resultA.txt", "r")
    data_y = np.array([])
    count = 0
    for i,line in enumerate(file):
        count = count + 1
        y = np.array(line)
        data_y = np.append(data_y, y)
    data_y = data_y.reshape(count, 1)
    return data_y.astype(float)


if __name__ == '__main__':
    data_x = data_x()
    data_y = data_y()
    # print(data_x)
    # print(data_y)

    w = [0, 0]
    b = 0
    alpha = 0.01
    for i in range(500):
        # print("AAAAAAA     ",w[0], w[1], b)
        for j,obj in enumerate(data_x):
            x1,x2 = obj[0], obj[1]
            y_in = w[0]*x1+w[1]*x2+b
            # print(data_y[j] - y_in)
            # print(w[0], w[1], b)
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