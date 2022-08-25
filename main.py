import numpy as np
import matplotlib.pyplot as plt
import random


X_train = np.loadtxt('train_X.csv', delimiter = ',').T
Y_train = np.loadtxt('train_label.csv', delimiter = ',').T
X_test = np.loadtxt('test_X.csv', delimiter = ',').T
Y_test = np.loadtxt('test_label.csv', delimiter = ',').T


print("shape of X_train :", X_train.shape)
print("shape of Y_train :", Y_train.shape)
print("shape of X_test :", X_test.shape)
print("shape of Y_test :", Y_test.shape)


class Neurol_network():
    def __init__(self,learning_rate,iterations,x ,n_h, y):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.n_x = x.shape[0]
        self.n_y = y.shape[0]
        self.n_h = n_h
        self.x = x
        self.y = y
        self.w1 = np.random.randn(self.n_h, self.n_x)*0.01
        self.b1 = np.zeros((self.n_h, 1))
        self.w2 = np.random.randn(self.n_y, self.n_h)*0.01
        self.b2 = np.zeros((self.n_y, 1))


    def fit(self):
           
        for i in range(self.iterations):
            z1 = np.dot(self.w1, self.x) + self.b1
            a1 = self.tanh(z1)
            z2 = np.dot(self.w2, a1) + self.b2
            a2 = self.softmax(z2)

            m = self.x.shape[1]
    
            dz2 = (a2 - self.y)
            dw2 = (1/m)*np.dot(dz2, a1.T)
            db2 = (1/m)*np.sum(dz2, axis = 1, keepdims = True)
            
            dz1 = (1/m)*np.dot(self.w2.T, dz2) * self.derivative_tanh(a1)
            dw1 = (1/m)*np.dot(dz1, self.x.T)
            db1 = (1/m)*np.sum(dz1, axis = 1, keepdims = True)

            self.w1 = self.w1 - self.learning_rate*dw1
            self.b1 = self.b1 - self.learning_rate*db1
            self.w2 = self.w2 - self.learning_rate*dw2
            self.b2 = self.b2 - self.learning_rate*db2

    def predict(self):
     
        idx = int(random.randrange(0,X_test.shape[1]))

        plt.imshow(X_test[:, idx].reshape((28,28)),cmap='gray')
        plt.show()

        z1 = np.dot(self.w1,X_test) + self.b1
        a1 = self.tanh(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = self.softmax(z2)

        a_pred = np.argmax(a2, 0)
        print(a_pred[idx])

    def tanh(self,x):
       return np.tanh(x)

    def relu(self,x):
       return np.maximum(x, 0)

    def softmax(self,x):
       expX = np.exp(x)
       return expX/np.sum(expX, axis = 0)


    def derivative_tanh(self,x):
       return (1 - np.power(np.tanh(x), 2))

    def derivative_relu(self,x):
       return np.array(x > 0, dtype = np.float32)  

    

test = Neurol_network(0.02,100,X_train,1000,Y_train)
test.fit()
test.predict()
test.predict()
test.predict()
test.predict()
test.predict()
test.predict()
test.predict()
test.predict()



