# A Script to test concepts
from time import time
import numpy as np
from ML.PCA import prince
from ML.mean_normalization import meannorm
from ML.neural_network import NeuralNet
from scipy import misc
from ML.k_means import kmeans
from ML.one_hot import one_hot

# 784 column in mnist train set, 28 x 28
filename = r'Machine-Learning\Test Data\MNIST\mnist_test.csv'
with open(filename, newline='') as csvfile:
    valuesx = np.matrix(np.loadtxt(csvfile, delimiter=",", usecols=range(1, 785)))
with open(filename, newline='') as csvfile:
    valuesy = np.matrix(np.loadtxt(csvfile, delimiter=",", usecols=(0,)))

# Transpose vector from [1xn] to [mx1]
valuesy = valuesy.T

# Convert from interger to one hot arrays
valuesy = one_hot(valuesy, 10)

valuesz = valuesx

# valuesz = prince(valuesz)
# valuesz = meannorm(valuesz)

# clusters = 10
# timenow = time()
# valuesy = one_hot(kmeans(valuesz, clusters), clusters)
# print(time() - timenow)

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
