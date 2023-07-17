import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.linear_model
from sklearn.metrics import confusion_matrix, roc_curve, RocCurveDisplay

import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import math
from tqdm import tqdm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay





def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def logistic(s):
    return 1.0 / (1.0 + np.exp(-s))

def simoid(s):
	return s * (1 - s)






class Linear:
    def __init__(self, neurony, wejscia):
        self.m_w = np.random.randn(wejscia, neurony)
        self.v_b = np.zeros((1, neurony))
        self.v_X = None

    def forward(self, v_x):
        self.v_X = v_x

        return (self.v_X @ self.m_w) + self.v_b

    def backward(self, blad, eta):
        self.m_w -= eta * (self.v_X.T @ blad)
        self.v_b -= eta * blad
        return blad @ self.m_w.T

class Activation:
    def __init__(self):
        self.s = None

    def forward(self,s):
        self.s = s

        return logistic(self.s)

    def backward(self, blad,eta):
        return simoid(logistic(self.s))*blad

class Layer:
    def __init__(self):
        self.warstwa = []

    def nowa_warstwa(self,neurony,wejscia):
        self.warstwa.append(Linear(neurony,wejscia))
        self.warstwa.append(Activation())
    def learn(self,epoki,X,Y):
        for a in tqdm(range(epoki)):
            for i in range(len(X)):
                wyjscie = X[i].reshape(1, -1)
                for j in self.warstwa:
                    wyjscie = j.forward(wyjscie)

                blad = 2 * (wyjscie - Y[i])/Y.size


                for j in reversed(self.warstwa):
                    blad = j.backward(blad, eta=0.1)

    def predict(self, X):



        # for i in range(len(X)):
        #     wyjscie = X[i].reshape(1, -1)
        #     for j in self.warstwa:
        #         wyjscie = j.forward(wyjscie)

        result = X
        print(result, 'asdasd')
        wyjsciee = []
        for sample in range(len(result)):
            wyjscie = result[sample].reshape(1, -1)
            for layer in self.warstwa:
                wyjscie = layer.forward(wyjscie)
            wyjsciee.append(wyjscie)
        return np.where(np.array(wyjsciee) > 0.5, 1, 0)

