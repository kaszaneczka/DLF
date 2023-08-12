import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.linear_model
from sklearn.metrics import confusion_matrix, roc_curve, RocCurveDisplay

from math import sqrt
from numpy import mean
from numpy.random import rand

import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import math
from tqdm import tqdm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
mse = tf.keras.losses.MeanSquaredError()
def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))

def sigmoid_der(s):
	return s * (1 - s)#np.exp(-s) / ((1.0 + np.exp(-s))**2)#

def relu(s):
    return np.maximum(s,0)

def relu_der(s):
    return np.where(s > 0, 1, 0)
def softmax(s):
    #print(s,'dddddd')
    return (np.exp(s)/(np.sum(np.exp(s))))


class Model:
    def __init__(self, input_shape: tuple):
        self.warstwa = []
        self.input_shape = input_shape
        self.linearr = input_shape[0]*input_shape[1]*input_shape[2]

    def linear(self,neurony):
        self.warstwa.append(Linear(neurony,self.linearr))
        self.linearr = neurony

    def conv(self,channels,kernel_shape,stride = (1,1), padding = (0,0)):
        self.warstwa.append(Conv(channels,kernel_shape,self.input_shape,stride = stride))
        # obliczanie kształtu na wyjsciu warstwy
        self.input_shape = (channels, ((self.input_shape[0]-kernel_shape[0]+2*padding[0])/stride[0])+1,
                            ((self.input_shape[1]-kernel_shape[0]+2*padding[1])/stride[1])+1)
        # obliczanie ilosci wejść do warstwy liniowej
        self.linearr = int(self.input_shape[0]*self.input_shape[1]*self.input_shape[2])
        print(self.linearr)

    def sigmoid(self):
        self.warstwa.append(Sigmoid())

    def relu(self):
        self.warstwa.append(Relu())

    def softmax(self):
        self.warstwa.append(Softmax())

    def flatten(self):
        self.warstwa.append(Flatten())




    def learn(self,epoki,X,Y,batch = 1,eta = 0.1):
        for a in tqdm(range(epoki)):
            for i in range(int(len(X)/batch)):
                #print(X.shape, 'asdasdasdasdasd')
                wyjscie = X[i*batch:i*batch+batch]#.reshape(batch,X.shape[1]*X.shape[2])#.flatten()
                #print(wyjscie.shape)

                for j in self.warstwa:
                    wyjscie = j.forward(wyjscie)
                #MSE
                blad = 2 * (wyjscie - (np.array(Y[i*batch:i*batch+batch])) )/np.array([Y[i*batch:i*batch+batch]]).size

                #CCE
                #blad = (wyjscie - (np.array(Y[i*batch:i*batch+batch])))
                #print(blad)
                for j in reversed(self.warstwa):
                    blad = j.backward(blad, eta=eta)


            # display of error
            wyjscie = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
            for j in self.warstwa:
                wyjscie = j.forward(wyjscie)

            print(Y.shape, wyjscie.shape,Y.size)
            print( 'bład : ',((np.sum((wyjscie - Y) ** 2))) / Y.size )
            print(-np.mean(Y * np.log(wyjscie) + (1 - Y) * np.log(1 - wyjscie)))
            #blad1 = (np.sum((np.array([Y[i * batch:i * batch + batch]]).T - wyjscie) ** 2)) / batch
            #print(blad1)

    def predict(self, X):
        result = X
        #print(result, 'asdasd')
        wyjsciee = []

        for sample in range(len(result)):

            wyjscie = result[sample].reshape(1,X.shape[1]*X.shape[2])
            #print(wyjscie.shape)
            for layer in self.warstwa:
                wyjscie = layer.forward(wyjscie)
            wyjsciee.append(wyjscie)
        return np.array(wyjsciee) #np.where(np.array(wyjsciee) > 0.5, 1, 0)




class Linear:
    def __init__(self, neurony, wejscia):
        # self.m_w = np.random.randn(wejscia, neurony)
        # self.m_w = normalize(self.m_w)

        # calculate the range for the weights
        lower, upper = -(1.0 / sqrt(wejscia)), (1.0 / sqrt(wejscia))
        # generate random numbers
        numbers = rand(neurony*wejscia)
        # scale to the desired range
        scaled = np.array(lower + numbers * (upper - lower))
        self.m_w = scaled.reshape(wejscia, neurony)

        self.v_b = np.zeros((1, neurony))
        self.v_X = None

    def forward(self, v_x):
        self.v_X = v_x
        print(self.v_X.shape, self.m_w.shape)
        #print(((self.v_X @ self.m_w) + self.v_b).shape)
        return (self.v_X @ self.m_w) + self.v_b


    def backward(self, blad, eta):
        #print(self.v_X.T.shape , blad.shape)
        ll = self.m_w.T
        self.m_w -= eta * (self.v_X.T @ blad)
        #print(blad,self.v_b,self.v_b - np.sum(blad),np.sum(blad), 'asdasdasdasdasd')
        self.v_b -= eta * np.sum(blad)#blad
        #print(blad)
        return blad @ ll# self.m_w.T


class Conv:

    def __init__(self, number_of_kernels: int, kernel_shape: tuple, input_shape: tuple, stride=(1, 1)):
        self.stride = stride  # default (1,1)
        self.kernel_shape = (kernel_shape[0], kernel_shape[1], input_shape[2])
        self.kernels = np.array(
            [add_kernel(kernel_shape, input_shape) for a in range(number_of_kernels)])  # creating random kernels
        # define shape of kernel

        # print(self.kernels, self.kernels.shape)

    def forward(self, v_x):
        self.v_x = v_x.transpose(1,2,0)
        output = np.array([conv_channel(self.v_x, kernel) for kernel in self.kernels])

        output = output.transpose(1, 2, 0)
        return output

    def backward(self, blad, eta):
        ll = self.kernels
        print("asdasdasd",blad.shape)

        return 0


def add_kernel(kernel_shape, latent_shape):
    kernel_shape = (kernel_shape[0], kernel_shape[1], latent_shape[2])

    # calculate the range for the weights
    # lower, upper = -(sqrt(6.0) / sqrt(n + m)), (sqrt(6.0) / sqrt(n + m))
    lower, upper = -(1.0 / sqrt(latent_shape[0] * latent_shape[1])), (1.0 / sqrt(latent_shape[0] * latent_shape[1]))
    # generate random numbers
    numbers = rand(kernel_shape[0] * kernel_shape[1] * latent_shape[2])
    # scale to the desired range
    scaled = np.array(lower + numbers * (upper - lower))
    m_w = scaled.reshape(kernel_shape)
    return m_w


def conv_step(latent: np.array, mask: np.array):  # confirmed
    # correction test below
    # a = np.array([[[1,3],[2,4],[1,1]],[[3,1],[2,2],[1,1]],[[2,3],[3,1],[1,4]]])
    # b = np.array([[[1,2],[2,2]],[[3,2],[4,2]]])
    # print(a,b)
    # print(conv_step(a[:2,:2],b)) # should return 42 in this case
    # (calculations for dimensions = input ->(3,3,2), filter->(2,2,2))

    product = latent * mask
    return np.sum(product)


def conv_channel(v_x, kernel):
    # new_channel = []
    # for length in range(v_x.shape[1] - kernel.shape[1] + 1):
    #     new_channel_l = []
    #     for height in range(v_x.shape[0] - kernel.shape[0] + 1):
    #         new_channel_l.append(conv_step(v_x[length:length+kernel.shape[1],height:height+kernel.shape[0]],kernel))
    #
    #     new_channel.append(new_channel_l)

    new_channel = [[conv_step(v_x[length:length + kernel.shape[1], height:height + kernel.shape[0]], kernel)
                    for height in range(v_x.shape[0] - kernel.shape[0] + 1)]
                   for length in range(v_x.shape[1] - kernel.shape[1] + 1)]
    return np.array(new_channel)



class Sigmoid:
    def __init__(self):
        self.s = None

    def forward(self,s):
        self.s = sigmoid(s)
        #self.s = softmax(s)
        return self.s

    def backward(self, blad,eta):
        return (sigmoid_der(self.s))*blad

class Relu:
    def __init__(self):
        self.s = None

    def forward(self,s):
        self.s = relu(s)
        #print(self.s.shape)
        #self.s = softmax(s)
        return self.s

    def backward(self, blad,eta):
        return (relu_der(self.s))*blad

class Softmax:
    def __init__(self):
        self.s = None

    def forward(self,s):
        self.s = softmax(s)
        return self.s

    def backward(self, blad,eta):
        return 1*blad

class Flatten:
    def __init__(self):
        self.s_shape = None

    def forward(self,s):

        self.s_shape = s.shape
        #print(s.reshape(self.s_shape[2], -1).shape,'flatten')
        return s.reshape(self.s_shape[2], -1)

    def backward(self,s,eta):
        return s.reshape(self.s_shape)

