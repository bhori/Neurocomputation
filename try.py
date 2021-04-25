class CustomAdaline(object):

    

    def __init__(self, n_iterations=100, random_state=1, learning_rate=0.01):

        self.n_iterations = n_iterations

        self.random_state = random_state

        self.learning_rate = learning_rate

    '''
9
    Batch Gradient Descent 
10
    
11
    1. Weights are updated considering all training examples.
12
    2. Learning of weights can continue for multiple iterations
13
    3. Learning rate needs to be defined
14
    '''

    def fit(self, X, y):

        rgen = np.random.RandomState(self.random_state)

        self.coef_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for _ in range(self.n_iterations):

              activation_function_output = self.activation_function(self.net_input(X))

              errors = y - activation_function_output

              self.coef_[1:] = self.coef_[1:] + self.learning_rate*X.T.dot(errors)

              self.coef_[0] = self.coef_[0] + self.learning_rate*errors.sum() 

    

    '''
25
    Net Input is sum of weighted input signals
26
    '''
    def net_input(self, X):

            weighted_sum = np.dot(X, self.coef_[1:]) + self.coef_[0]

            return weighted_sum

    

    '''
32
    Activation function is fed the net input. As the activation function is
33
    an identity function, the output from activation function is same as the
34
    input to the function.
35
    '''

    def activation_function(self, X):

            return X

    

    '''
40
    Prediction is made on the basis of output of activation function
41
    '''

    def predict(self, X):

        return np.where(self.activation_function(self.net_input(X)) >= 0.0, 1, 0) 

    

    '''
46
    Model score is calculated based on comparison of 
47
    expected value and predicted value
48
    '''

    def score(self, X, y):

        misclassified_data_count = 0

        for xi, target in zip(X, y):

            output = self.predict(xi)

            if(target != output):

                misclassified_data_count += 1

        total_data_count = len(X)

        self.score_ = (total_data_count - misclassified_data_count)/total_data_count

        return self.score_
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
bc = datasets.load_breast_cancer()

X = bc.data

y = bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

adaline = CustomAdaline(n_iterations = 10)

adaline.fit(X_train, y_train)

adaline.score(X_test, y_test), prcptrn.score(X_train, y_train)