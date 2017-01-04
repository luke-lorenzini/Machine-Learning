# Receive a matrix [mx1] representing initial values
def NeuralNet(x, y):
    import math
    import numpy
    import numpy.matlib

    myx = x.T
    myy = y.T

    alpha = 0.001

    # Initialize a matrix of dimensions [mxn]
    # with random numbers
    # input: row = number of rows, col = number of columns
    # output: theta [mxn] matrix of random
    def init_weights(row, col):
        theta = numpy.matlib.rand(row, col)
        # theta = numpy.ones((row, col))
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
        # z = theta * a
        z = numpy.dot(theta, a)
        return z

    # Calculate the function g(z)
    # g(z) = 1/(1+e^(-z))
    # input: z = [nx1]
    # output a = [1x1]
    def calc_a(z):
        a = z
        # For each row in z, calculate the function g(z)
        # for i in range(z.shape[0]):
        #     a[i] = 1 / (1 + math.exp(-1 * z[i]))
        a = 1 / (1 + numpy.exp(-1 * z))
        return a

    # z represents network architecture (w/o biases)
    # need to modify third value to not be tied, should be able to handle anything
    arch = (myx.shape[0], 2 * myx.shape[0], myy.shape[0], myy.shape[0])

    ### Initialization ###
    # Use an [mxn] matrix to create an [nx(n+1)]
    theta1 = init_weights(arch[1], arch[0] + 1)
    # print("theta1")
    # print(theta1)

    theta2 = init_weights(arch[2], arch[1] + 1)
    # print("theta2")
    # print(theta2)

    for i in range(10000):
        # Same dimensionality as theta1 with an additional row for bias
        delta1 = numpy.zeros((theta1.shape[0] + 1, theta1.shape[1]))
        # print("delta1")
        # print(delta1)

        # Same dimensionality as theta2 without an additional row for bias (last layer)
        delta2 = numpy.zeros((theta2.shape[0], theta2.shape[1]))
        # print("delta2")
        # print(delta2)

        # m is a placeholder for the number of training examples
        # Each column represents a training example
        m = myx.shape[1]
        for j in range(m):
            ### Forward Propogation ###
            # Step 1, append the bias
            a1 = append(myx[:, j])
            # print("a1")
            # print(a1)

            # Step 2, calculate z
            z1 = calc_z(a1, theta1)
            # print("z1")
            # print(z1)

            # Step 3, transform z
            a2 = calc_a(z1)
            # print("a2")
            # print(a2)

            # Second layer
            # Step 1, append the bias
            a2 = append(a2)
            # print("a2")
            # print(a2)

            # Step 2, calculate z
            z2 = calc_z(a2, theta2)
            # print("z2")
            # print(z2)

            # Step 3, transform z
            a3 = calc_a(z2)
            # print("a3")
            # print(a3)

            ### Back Propogation ###
            # Calculate errors for layer 3
            d3 = a3 - myy[:, j]
            # print("d3")
            # print(d3)

            # Calculate errors for layer 2
            d2 = numpy.multiply((numpy.dot(theta2.T, d3)), (numpy.multiply(a2, (1 - a2))))
            # print("d2")
            # print(d2)

            # delta(l) = delta(l) + d(l+1)*a(l).T
            delta2 += numpy.dot(d3, a2.T)
            # print("delta2")
            # print(delta2)

            delta1 += numpy.dot(d2, a1.T)
            # print("delta1")
            # print(delta1)

        # outside of 1:m loop
        # theta(x), delta(x), and theta(x)_grad should have same dimensions
        theta2_grad = delta2 / m
        # print("theta2_grad")
        # print(theta2_grad)

        theta1_grad = delta1 / m
        # print("theta1_grad")
        # print(theta1_grad)

        # theta = theta - alpha * d/d_theta(J(theta))
        # Update theta
        theta2_t = theta2 - alpha * theta2_grad
        # print("theta2")
        # print(theta2)

        # print(abs(theta2 - theta2_t))
        numpy.copyto(theta2, theta2_t)

        # Remove the top row (all zeros anyway) from theta1_grad
        theta1_grad = numpy.delete(theta1_grad, 0, 0)

        theta1_t = theta1 - alpha * theta1_grad
        # print("theta1")
        # print(theta1)

        numpy.copyto(theta1, theta1_t)

    print("theta1")
    print(theta1)
    print("theta2")
    print(theta2)

    check1 = myx[:, 1]
    check1 = append(check1)
    check2 = calc_z(check1, theta1)
    check2 = calc_a(check2)
    
    check3 = append(check2)
    check3 = calc_z(check3, theta2)
    check3 = calc_a(check3)
    print(check3)
