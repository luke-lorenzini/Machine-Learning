# Receive a matrix [mx1] representing initial values
def NeuralNet(x, y):
    import math
    import numpy
    import numpy.matlib

    # Initialize a matrix of dimensions [mxn]
    # with random numbers
    # input: row = number of rows, col = number of columns
    # output: theta [mxn] matrix of random
    def init_weights(row, col):
        theta = numpy.matlib.rand(row, col)
        return theta

    # Append a '1' to the beginning of the vector
    # input: x = [nx1]
    # output x = [(n+1)x1]
    def append(x):
        theta = numpy.vstack((1, x))
        return theta

    # Solve z = theta * a
    # input: theta = [mxn], a = [nx1]
    # output: z = [mx1]
    def calc_z(a, theta):
        z = theta * a
        return z

    # Calculate the function g(z)
    # g(z) = 1/(1+e^(-z))
    # input: z = [nx1]
    # output a = [1x1]
    def calc_a(z):
        a = z
        # For each row in z, calculate the function g(z)
        for i in range(z.shape[0]):
            a[i] = 1 / (1 + math.exp(-1 * z[i]))
        return a

    # Use an [mxn] matrix to create an [nx(n+1)]
    theta1 = init_weights(x.shape[0], x.shape[0] + 1)
    print("theta 1")
    print(theta1)
    print("")

    theta2 = init_weights(3, 5)
    print("theta 2")
    print(theta2)
    print("")

    # Step 1, append the bias
    a1 = append(x)
    print("a1")
    print(a1)
    print("")

    # Step 2, calculate z
    z1 = calc_z(a1, theta1)
    print("z1")
    print(z1)
    print("")

    # Step 3, transform z
    a2 = calc_a(z1)
    print("a2")
    print(a2)
    print("")

    # second layer
    # Step 1, append the bias
    a2 = append(a2)
    print("a2")
    print(a2)
    print("")

    # Step 2, calculate z
    z2 = calc_z(a2, theta2)
    print("z2")
    print(z2)
    print("")

    # Step 3, transform z
    a3 = calc_a(z2)
    print("a3")
    print(a3)
    print("")

    # Start back propogation
    # Calculate errors for layer 3
    d3 = y - a3
    print("d3")
    print(d3)
    print("")

    # Calculate errors for layer 2
    d2 = numpy.multiply((theta2.T * d3), (numpy.multiply(a2, (1 - a2))))
    print("d2")
    print(d2)
    print("")
