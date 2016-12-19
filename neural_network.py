def NeuralNet(x):
    import math
    import numpy

    theta = 1

    # Append a '1' to the beginning of the vector
    # input: x = [nx1]
    # output x = [(n+1)x1]
    def append(x):
        #x = numpy.r_[np.ones(myx.shape[0]), myx]
        # myx =  numpy.r_["1", x]
        myx = x
        myx = numpy.vstack('1', x)
        return myx

    # Solve z = theta * a
    # input: theta = [mxn], a = [nx1]
    # output: z = [mx1]
    def calc_z(a, theta):
        return theta * a

    # calculate the function g(z)
    # g(z) = 1/(1+e^(-z))
    # input: z = [nx1]
    # output a = [1x1]
    def calc_a(z):
        a = z
        # For each row in z, calculate the function g(z)
        for i in range(z.shape[0]):
            a[i] = 1 / (1 + math.exp(-1 * z[i]))
        return a

    # Run sequence of append, calc_z, calc_a for L-1 layers in network
    #def fwdProp():
    #print(append(x))
    print(calc_a(x))
        #print(calc_a(calc_z(append(x), theta)))

