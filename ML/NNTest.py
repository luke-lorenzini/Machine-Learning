# A Script to test concepts
from time import time
import numpy as np
from  PCA import prince
from mean_normalization import meannorm
# from NNet import NeuralNet
from neural_network import NeuralNet
from scipy import misc
import wave

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

filename = r'data\banknote\banknote.csv'
with open(filename, newline='') as csvfile:
    valuesx = np.matrix(np.loadtxt(csvfile, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3)))
with open(filename, newline='') as csvfile:
    valuesy = np.matrix(np.loadtxt(csvfile, delimiter=",", skiprows=1, usecols=(4,)))

filename = r'data\iris\irisNN.csv'
with open(filename, newline='') as csvfile:
    valuesx = np.matrix(np.loadtxt(csvfile, delimiter=",", usecols=(0, 1, 2, 3)))
with open(filename, newline='') as csvfile:
    valuesy = np.matrix(np.loadtxt(csvfile, delimiter=",", usecols=(5, 6, 7)))

# 784 column in mnist train set, 28 x 28
# filename = r'C:\Users\Luke Lorenzini\Documents\MNIST\mnist_test.csv'
filename = r'data\MNIST\mnist_train.csv'
with open(filename, newline='') as csvfile:
    valuesx = np.matrix(np.loadtxt(csvfile, delimiter=",", usecols=range(1, 785)))
with open(filename, newline='') as csvfile:
    valuesy = np.matrix(np.loadtxt(csvfile, delimiter=",", usecols=(0,)))

# Read a wav file, convert it to L & R channels
# filename = r'data\sound\folk.wav'
# ff = wave.open(filename, 'rb')
# valuesx = np.zeros((ff.getnframes(), ff.getnchannels()))
# valuesx = np.matrix(np.zeros((ff.getnframes(), ff.getnchannels())))
# for x in range(ff.getnframes()):
#     zz = ff.readframes(1)
#     valuesx[x, 0] = (zz[0] << 16) + (zz[1] << 8) + (zz[2])
#     valuesx[x, 1] = (zz[3] << 16) + (zz[4] << 8) + (zz[5])

# valuesy = valuesx[1:valuesx.shape[0], :]
# valuesx = valuesx[0:valuesx.shape[0] - 1, :]

# Use these when only viewing one column from dataset
if valuesx.shape[0] == 1:
    valuesx = valuesx.T
if valuesy.shape[0] == 1:
    valuesy = valuesy.T

valuesy = onehot_mnist(valuesy)

# valuesx = np.matrix([[0, 0], [0, 1], [1, 0], [1, 1]])
# valuesy = np.matrix([[0], [1], [1], [0]])

valuesz = valuesx

valuesz = prince(valuesz)
valuesz = meannorm(valuesz)
# valuesy = meannorm(valuesy)

timenow = time()
t1, t2 = NeuralNet(valuesz, valuesy)
print(time() - timenow)

np.savetxt("theta1.csv", t1, delimiter=",")
np.savetxt("theta2.csv", t2, delimiter=",")

zero = np.matrix([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
first = t2.T * zero
second = t1.T * first
third = np.reshape(second, (-1, 28))
misc.imsave('test0.png', third) # uses the Image module (PIL)

one = np.matrix([[0], [1], [0], [0], [0], [0], [0], [0], [0], [0]])
first = t2.T * one
second = t1.T * first
third = np.reshape(second, (-1, 28))
misc.imsave('test1.png', third) # uses the Image module (PIL)

two = np.matrix([[0], [0], [1], [0], [0], [0], [0], [0], [0], [0]])
first = t2.T * two
second = t1.T * first
third = np.reshape(second, (-1, 28))
misc.imsave('test2.png', third) # uses the Image module (PIL)

three = np.matrix([[0], [0], [0], [1], [0], [0], [0], [0], [0], [0]])
first = t2.T * three
second = t1.T * first
third = np.reshape(second, (-1, 28))
misc.imsave('test3.png', third) # uses the Image module (PIL)

four = np.matrix([[0], [0], [0], [0], [1], [0], [0], [0], [0], [0]])
first = t2.T * four
second = t1.T * first
third = np.reshape(second, (-1, 28))
misc.imsave('test4.png', third) # uses the Image module (PIL)

five = np.matrix([[0], [0], [0], [0], [0], [1], [0], [0], [0], [0]])
first = t2.T * five
second = t1.T * first
third = np.reshape(second, (-1, 28))
misc.imsave('test5.png', third) # uses the Image module (PIL)

six = np.matrix([[0], [0], [0], [0], [0], [0], [1], [0], [0], [0]])
first = t2.T * six
second = t1.T * first
third = np.reshape(second, (-1, 28))
misc.imsave('test6.png', third) # uses the Image module (PIL)

seven = np.matrix([[0], [0], [0], [0], [0], [0], [0], [1], [0], [0]])
first = t2.T * seven
second = t1.T * first
third = np.reshape(second, (-1, 28))
misc.imsave('test7.png', third) # uses the Image module (PIL)

eight = np.matrix([[0], [0], [0], [0], [0], [0], [0], [0], [1], [0]])
first = t2.T * eight
second = t1.T * first
third = np.reshape(second, (-1, 28))
misc.imsave('test8.png', third) # uses the Image module (PIL)

nine = np.matrix([[0], [0], [0], [0], [0], [0], [0], [0], [0], [1]])
first = t2.T * nine
second = t1.T * first
third = np.reshape(second, (-1, 28))
misc.imsave('test9.png', third) # uses the Image module (PIL)
