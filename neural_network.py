# Receive a matrix [mx1] representing initial values
def NeuralNet(x, y):
    import math
    import numpy
    import numpy.matlib

    # z represents network architecture (no biases)
    arch = (x.shape[0], 4, 3, y.shape[0])
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

    ### Initialization ###
    # Use an [mxn] matrix to create an [nx(n+1)]
    theta1 = init_weights(arch[1], x.shape[0] + 1)
    # print("theta1")
    # print(theta1)
    # print("")

    delta1 = numpy.zeros((5, 5))
    # print("delta1")
    # print(delta1)
    # print("")

    theta2 = init_weights(arch[2], 5)
    # print("theta2")
    # print(theta2)
    # print("")
    
    delta2 = numpy.zeros((3, 5))
    # print("delta2")
    # print(delta2)
    # print("")

    # m is a placeholder for the number of training examples
    m = 1
    for i in range(500):
        for j in range(m):
            ### Forward Propogation ###
            # Step 1, append the bias
            a1 = append(x)
            # print("a1")
            # print(a1)
            # print("")

            # Step 2, calculate z
            z1 = calc_z(a1, theta1)
            # print("z1")
            # print(z1)
            # print("")

            # Step 3, transform z
            a2 = calc_a(z1)
            # print("a2")
            # print(a2)
            # print("")

            # second layer
            # Step 1, append the bias
            a2 = append(a2)
            # print("a2")
            # print(a2)
            # print("")

            # Step 2, calculate z
            z2 = calc_z(a2, theta2)
            # print("z2")
            # print(z2)
            # print("")

            # Step 3, transform z
            a3 = calc_a(z2)
            # print("a3")
            # print(a3)
            # print("")

            ### Back Propogation ###
            # Calculate errors for layer 3
            d3 = y - a3
            # print("d3")
            # print(d3)
            # print("")

            # Calculate errors for layer 2
            d2 = numpy.multiply((theta2.T * d3), (numpy.multiply(a2, (1 - a2))))
            # print("d2")
            # print(d2)
            # print("")

            # delta(l) = delta(l) + d(l+1)*a(l).T
            # delta2 = delta2 + d3 * a2.T
            delta2 += numpy.dot(d3, a2.T)
            # print("delta2")
            # print(delta2)
            # print("")

            # delta1 = delta1 + d2 * a1.T
            delta1 += numpy.dot(d2, a1.T)
            # print("delta1")
            # print(delta1)
            # print("")

        # outside of 1:m loop
        # theta(x), delta(x), and theta(x)_grad should have same dimensions
        theta2_grad = delta2 / m
        # print("theta2_grad")
        # print(theta2_grad)
        # print("")

        theta1_grad = delta1 / m
        # print("theta1_grad")
        # print(theta1_grad)
        # print("")

        # theta = theta - alpha * d/d_theta(J(theta))
        # Update theta
        theta2_t = theta2 - alpha * theta2_grad
        # print("theta2")
        # print(theta2)
        # print("")

        # print(abs(theta2 - theta2_t))
        numpy.copyto(theta2, theta2_t)

        # Remove the top row (all zeros anyway) from theta1_grad
        theta1_grad = numpy.delete(theta1_grad, 0, 0)

        theta1_t = theta1 - alpha * theta1_grad
        # print("theta1")
        #print(theta1)
        # print("")

        difference1 = abs(theta1 - theta1_t)
        print(difference1)
        numpy.copyto(theta1, theta1_t)

    print(theta1)
