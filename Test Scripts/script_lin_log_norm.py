from time import time
import numpy as np
from linear_regression import linreg
from logistic_regression import logreg
from normal import norm

filename = r'data\iris\Iris-versicolor.csv'

#myx = np.matrix([[0], [1], [2], [3]])
#myy = np.matrix([[1], [3], [5], [7]])

# with open('data\winequality-red.csv', newline='') as csvfile:
#     myx = np.matrix(np.loadtxt(csvfile, delimiter=";", skiprows=1, usecols=(0, 1)))
# with open('data\winequality-red.csv', newline='') as csvfile:
#     myy = np.matrix(np.loadtxt(csvfile, delimiter=";", skiprows=1, usecols=(11,)))

with open(filename, newline='') as csvfile:
    myx = np.matrix(np.loadtxt(csvfile, delimiter=",", usecols=(0, )))
with open(filename, newline='') as csvfile:
    myy = np.matrix(np.loadtxt(csvfile, delimiter=",", usecols=(4,)))

if (myx.shape[0] == 1):
    myx = myx.T
if (myy.shape[0] == 1):
    myy = myy.T
# Append a column of ones to X
myx = np.c_[np.ones(myx.shape[0]), myx]

theta = np.matrix(np.ones(myx.shape[1]))
alpha = 0.001

# timenow = time()
# print(linreg(myx, myy, theta, alpha, 0))
# print(time() - timenow)
# print("")

timenow = time()
print(logreg(myx, myy, theta, alpha, 2))
print(time() - timenow)
print("")

# timenow = time()
# print(norm(myx, myy))
# print(time() - timenow)
# print("")
