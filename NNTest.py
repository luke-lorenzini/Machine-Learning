# A Script to test concepts
from time import time
import numpy as np
from  PCA import prince
from mean_normalization import meannorm
from NNet import NeuralNet
# from neural_network3 import NeuralNet

def onehot_mnist(oneHot):
    # Convert from integer to one hot, crude, but functional
    newy = np.matrix(np.zeros((oneHot.shape[0], 10)))
    for i in range(oneHot.shape[0]):
        if (oneHot[i, 0] == 0):
            newy[i] = np.matrix([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif (oneHot[i, 0] == 1):
            newy[i] = np.matrix([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif (oneHot[i, 0] == 2):
            newy[i] = np.matrix([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif (oneHot[i, 0] == 3):
            newy[i] = np.matrix([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif (oneHot[i, 0] == 4):
            newy[i] = np.matrix([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif (oneHot[i, 0] == 5):
            newy[i] = np.matrix([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        elif (oneHot[i, 0] == 6):
            newy[i] = np.matrix([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif (oneHot[i, 0] == 7):
            newy[i] = np.matrix([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif (oneHot[i, 0] == 8):
            newy[i] = np.matrix([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif (oneHot[i, 0] == 9):
            newy[i] = np.matrix([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    return newy

filename = r'data\iris\irisNN.csv'
with open(filename, newline='') as csvfile:
    valuesx = np.matrix(np.loadtxt(csvfile, delimiter=",", usecols=(0, 1, 2, 3)))
with open(filename, newline='') as csvfile:
    valuesy = np.matrix(np.loadtxt(csvfile, delimiter=",", usecols=(5, 6, 7)))

# filename = r'data\banknote\banknote.csv'
# with open(filename, newline='') as csvfile:
#     valuesx = np.matrix(np.loadtxt(csvfile, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3)))
# with open(filename, newline='') as csvfile:
#     valuesy = np.matrix(np.loadtxt(csvfile, delimiter=",", skiprows=1, usecols=(4,)))

# 784 column in mnist train set, 28 x 28
# filename = r'data\MNIST\mnist_train.csv'
# with open(filename, newline='') as csvfile:
#     valuesx = np.matrix(np.loadtxt(csvfile, delimiter=",", usecols=range(1, 785)))
# with open(filename, newline='') as csvfile:
#     valuesy = np.matrix(np.loadtxt(csvfile, delimiter=",", usecols=(0,)))

# Use these when only viewing one column from dataset
if valuesx.shape[0] == 1:
    valuesx = valuesx.T
if valuesy.shape[0] == 1:
    valuesy = valuesy.T

# valuesy = onehot_mnist(valuesy)

valuesx = np.matrix([[0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]])
valuesy = np.matrix([[0], [1], [1], [0]])

# valuesx = np.matrix([[2104, 5, 1, 45], [1416, 3, 2, 40], [1534, 3, 2, 30], [852, 2, 1, 36]])
# valuesy = np.matrix([[460], [232], [315], [178]])

valuesz = valuesx

# TODO Feature Scaling
# Principle Component Analysis
valuesz = prince(valuesz)
# Mean Normalization
valuesz = meannorm(valuesz)


timenow = time()
t1, t2 = NeuralNet(valuesz, valuesy)
print(time() - timenow)

print(t1)
print(t2)
