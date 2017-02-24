# A Script to test concepts
from time import time
import numpy as np
from  PCA import prince
from mean_normalization import meannorm
from neural_network import NeuralNet
from k_means import kmeans
from one_hot import one_hot

filename = r'data\iris\irisNN.csv'
with open(filename, newline='') as csvfile:
    valuesx = np.matrix(np.loadtxt(csvfile, delimiter=",", usecols=(0, 1, 2, 3)))
with open(filename, newline='') as csvfile:
    valuesy = np.matrix(np.loadtxt(csvfile, delimiter=",", usecols=(5, 6, 7)))

valuesz = valuesx

valuesz = prince(valuesz)
valuesz = meannorm(valuesz)

clusters = 3
timenow = time()
valuesy = one_hot(kmeans(valuesz, clusters), clusters)
print(time() - timenow)

timenow = time()
t1, t2 = NeuralNet(valuesz, valuesy)
print(time() - timenow)

# np.savetxt("theta1.csv", t1, delimiter=",")
# np.savetxt("theta2.csv", t2, delimiter=",")
