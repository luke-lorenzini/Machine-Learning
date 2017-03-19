# A Script to test concepts
from time import time
import numpy as np
from ML.PCA import prince
from ML.mean_normalization import meannorm
from ML.neural_network import NeuralNet
from ML.k_means import kmeans

# The compression rate for PCA
rate = 99

filename = r'..\Machine-Learning\Test-Data\banknote\banknote.csv'
with open(filename, newline='') as csvfile:
    valuesx = np.matrix(np.loadtxt(csvfile, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3)))
with open(filename, newline='') as csvfile:
    valuesy = np.matrix(np.loadtxt(csvfile, delimiter=",", skiprows=1, usecols=(4,)))

# Transpose vector from [1xn] to [mx1]
valuesy = valuesy.T

valuesz = valuesx

valuesz = prince(valuesz, rate)
valuesz = meannorm(valuesz)

# kmeans(valuesz, 2)

timenow = time()
t1, t2 = NeuralNet(valuesz, valuesy)
print(time() - timenow)

np.savetxt("theta1.csv", t1, delimiter=",")
np.savetxt("theta2.csv", t2, delimiter=",")
