# Receive a matrix [mx1] representing initial values
def NeuralNet(x, y):
    import math
    import numpy
    import numpy.matlib

    # Set random seed for ease of testing
    numpy.random.seed(1)

    epochs = 10000
    lamb = 0.001
    myBias = 0

    myx = x.T
    myy = y.T

    test_myx = x
    test_myy = y

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
    def sigmoid(z):
        a = z
        # For each row in z, calculate the function g(z)
        a = 1 / (1 + numpy.exp(-1 * z))
        return a

    def sigmoid_prime(z):
        a = z
        a = numpy.multiply(z, (1 - z))
        return a

    def nonlin(x, deriv=False):
        if deriv:
            # return x*(1-x)
            return numpy.multiply(x, (1 - x))
        return 1 / (1+numpy.exp(-x))

    # z represents network architecture (w/o biases)
    # need to modify third value to not be tied, should be able to handle anything
    arch = (myx.shape[0], 1 * myx.shape[0], 1 * myy.shape[0], myy.shape[0])

    ### Initialization ###
    # Use an [mxn] matrix to create an [nx(n+1)]
    # theta1 = init_weights(arch[1], arch[0] + 1)
    theta1 = init_weights(arch[1] + 1, arch[0] + myBias)
    # print("theta1")
    # print(theta1)

    # theta2 = init_weights(arch[2], arch[1] + 1)
    theta2 = init_weights(arch[3], arch[1] + 1 + myBias)
    # print(theta2)

    # test_theta1 = init_weights(3, 4)
    # test_theta2 = init_weights(4, 1)

    test_theta1 = init_weights(arch[0], arch[1] + 1)
    test_theta2 = init_weights(arch[1] + 1, arch[3])

    for i in range(epochs):
        # Each column represents a training example
        ### Forward Propogation ###
        # Step 1, append the bias
        a1 = myx
        # a1 = append(myx)
        test_a1 = test_myx
        # print("a1")
        # print(a1)

        # Step 2, calculate z
        z1 = calc_z(a1, theta1)
        test_a2_a = nonlin(numpy.dot(test_a1, test_theta1))
        # print("z1")
        # print(z1)

        # Step 3, transform z
        a2 = sigmoid(z1)
        # test_a2 = nonlin(test_a2_a)
        # print("a2")
        # print(a2)

        # Second layer
        # Step 1, append the bias
        # a2 = append(a2)
        # print("a2")
        # print(a2)

        # Step 2, calculate z
        z2 = calc_z(a2, theta2)
        test_a3_a = nonlin(numpy.dot(test_a2_a, test_theta2))
        # print("z2")
        # print(z2)

        # Step 3, transform z
        a3 = sigmoid(z2)
        # test_a3 = nonlin(test_a3_a)
        # print("a3")
        # print(a3)

        ### Back Propogation ###
        # Calculate errors for layer 3
        d3_error = myy - a3
        test_d3_error = test_myy - test_a3_a
        if (i % (epochs / 10)) == 0:
            print(100 * numpy.mean(numpy.abs(d3_error)))
            print(100 * numpy.mean(numpy.abs(test_d3_error)))

        # d3_delta_a = (1 - a3)
        # d3_delta_b = numpy.multiply(a3, d3_delta_a)
        # d3_delta_c = numpy.multiply(d3_error, d3_delta_b)
        #print(d3_delta_c)
        d3_delta = numpy.multiply(d3_error, sigmoid_prime(a3))
        # print(d3_delta - d3_delta_c)
        test_d3_delta = numpy.multiply(test_d3_error, nonlin(test_a3_a, deriv=True))

        # Calculate errors for layer 2
        d2_error = numpy.dot(theta2.T, d3_delta)
        test_d2_error = test_d3_delta.dot(test_theta2.T)
        # print("d2")
        # print(d2)

        # d2_delta_a = (1 - a2)
        # d2_delta_b = numpy.multiply(a2, d2_delta_a)
        # d2_delta_c = numpy.multiply(d2_error, d2_delta_b)
        d2_delta = numpy.multiply(d2_error, sigmoid_prime(a2))
        # print(d2_delta - d2_delta_c)
        test_d2_delta = numpy.multiply(test_d2_error, nonlin(test_a2_a, deriv=True))

        # delta(l) = delta(l) + d(l+1)*a(l).T
        # delta2 += numpy.dot(d3_error, a2.T)
        theta2 += (numpy.dot(d3_delta, a2.T) + lamb * theta2)
        test_theta2 += test_a2_a.T.dot(test_d3_delta)
        # print("delta2")
        # print(delta2)

        # d2_delta = numpy.delete(d2_delta, 0, 0)
        theta1 += (numpy.dot(d2_delta, a1.T) + lamb * theta1)
        test_theta1 += test_a1.T.dot(test_d2_delta)
        # print("delta1")
        # print(delta1)

    # Check the results by analyzing one training example (in this case 0)
    check1 = myx
    # check1 = append(check1)
    check2 = calc_z(check1, theta1)
    check2 = sigmoid(check2)
    # check3 = append(check2)
    check3 = calc_z(check2, theta2)
    check3 = sigmoid(check3)
    print(100 * check3)

    return theta1, theta2
