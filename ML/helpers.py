import numpy
import numpy.matlib

# Initialize a matrix of dimensions [mxn]
# with random numbers
# input: row = number of rows, col = number of columns
# output: theta [mxn] matrix of random
def init_weights(row, col):
    theta = numpy.matlib.rand(row, col)
    theta = 2 * numpy.random.random((row, col)) - 1
    # theta = numpy.ones((row, col))
    return theta

# Append a '1' to the beginning of the vector
# input: x = [nx1]
# output x = [(n+1)x1]
def append(x):
    theta = numpy.vstack((numpy.ones(x.shape[1]), x))
    return theta

# Solve z = theta * a
# input: theta = [mxn], a = [nx1]
# output: z = [mx1]
def calc_z(a, theta):
    # z = theta * a
    z = numpy.dot(theta, a)
    return z

# Calculate the function g(z)
# g(z) = 1/(1+e^(-z))
# input: z = [nx1]
# output a = [1x1]
def sig(z, prime=False):
    a = z
    if prime:
        a = numpy.multiply(z, (1 - z))
    else:
        # For each row in z, calculate the function g(z)
        a = 1 / (1 + numpy.exp(-1 * z))
    return a

#https://en.wikipedia.org/wiki/Activation_function
def tanh(z, prime=False):
    a = z
    if prime:
        a = 1 - numpy.multiply(z, z)
    else:
        # For each row in z, calculate the function g(z)
        a = 2 / (1 + numpy.exp(-2 * z)) - 1
    return a

def relu(z, prime=False):
    a = z
    if prime:
        a = 1
    else:
        a = 1
    return a

def softmax(z, prime=False):
    # Not yet implemented, just a placeholder
    # a = z
    if prime:
        a = z
    else:
        a = z
    return a

def calc_alpha(i, const1, const2):
    return const1 / (i + const2)
        