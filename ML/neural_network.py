def NeuralNet(x, y):
    import numpy
    import numpy.matlib
    from ML.helpers import init_weights
    from ML.helpers import calc_alpha
    from ML.helpers import calc_z
    from ML.helpers import sig

    # mode = 0, batch gradient descent
    # mode = 1, vectorized batch gradient descent
    # mode = 2, stochastic gradient descent
    # mode = 3, mini-batch gradient descent
    mode = 1

    # Modify learning rate, alpha
    alpha = 1
    const1 = 1
    const2 = 5

    # Map Reduce
    # like batch gradient descent, divide number of
    # samples into N clusters, each 'machine' processes
    # a cluster concurrently n1, n2...nN, clusters
    # combined after,
    # theta -= alpha(n1 + n2 + ... + nN)

    # Set random seed for ease of testing
    numpy.random.seed(1)

    myx = x.T
    myy = y.T

    epochs = 1000
    samples = myx.shape[1]
    lamb = 0.001
    batchsize = 100

    error = 1

    # arch represents network architecture
    arch = (myx.shape[0], 2 * myx.shape[0], 2 * myy.shape[0], myy.shape[0])

    ### Initialization ###
    # Use an [mxn] matrix to create an [nx(n+1)]
    theta1 = init_weights(arch[1], arch[0])
    thetax = init_weights(arch[len(arch) - 1], arch[1])

    d1_delta = numpy.zeros((theta1.shape[0], theta1.shape[1]))
    dx_delta = numpy.zeros((thetax.shape[0], thetax.shape[1]))

    def fwd_prop(mat, theta):
        """Forward Propogation"""
        # Each column represents a training example

        # Step 2, calculate z
        z = calc_z(mat, theta)

        # Step 3, transform z
        act = sig(z)

        return act

    def back_prop(act, mat, err):
        """ Back Propogation"""
        d_err = numpy.multiply(numpy.dot(mat.T, err), sig(act, True))

        return d_err

    def calc_err(i, j, err):
        if (i % (epochs / 10)) == 0:
            if j == 0:
                error = numpy.mean(numpy.abs(err))
                print(100 * numpy.mean(numpy.abs(err)))

    for i in range(epochs):
    # i = 0
    # while (i < epochs) | (error < 1E-5):
    #     i += 1
        alpha = calc_alpha(i, const1, const2)
        if mode == 0: # batch gradient descent
            # Run forward and back prop on each sample, then update
            for j in range(samples):
                # Setup input / output data
                in_mat = myx[:, j]
                out_mat = myy[:, j]

                # Forward Propogation
                actvtn1 = in_mat
                actvtn2 = fwd_prop(actvtn1, theta1)
                actvtn3 = fwd_prop(actvtn2, thetax)

                # Back Propogation
                d3_error = actvtn3 - out_mat

                calc_err(i, j, d3_error)

                # d2_error = back_prop(actvtn2, thetax, d3_error)
                # d1_error = back_prop(actvtn1, theta1, d2_error)

                # dx_delta += numpy.dot(d3_error, actvtn2.T)
                # d1_delta += numpy.dot(d2_error, actvtn1.T)

                d2_error = numpy.multiply(d3_error, sig(actvtn3, True))
                d1_error = back_prop(actvtn2, thetax, d2_error)

                dx_delta += numpy.dot(d2_error, actvtn2.T)
                d1_delta += numpy.dot(d1_error, actvtn1.T)

            theta1 -= alpha * (d1_delta / samples + (lamb * theta1))
            thetax -= alpha * (dx_delta / samples + (lamb * thetax))

        elif mode == 1: # vectorized batch gradient descent
            # Calculate the whole shebang
            # Setup input / output data
            in_mat = myx
            out_mat = myy

            # Forward Propogation
            actvtn1 = in_mat
            actvtn2 = fwd_prop(actvtn1, theta1)
            actvtn3 = fwd_prop(actvtn2, thetax)

            # Back Propogation
            d3_error = actvtn3 - out_mat

            calc_err(i, 0, d3_error)

            # d2_error = back_prop(actvtn2, thetax, d3_error)
            # d1_error = back_prop(actvtn1, theta1, d2_error)

            # dx_delta += numpy.dot(d3_error, actvtn2.T)
            # d1_delta += numpy.dot(d2_error, actvtn1.T)

            d2_error = numpy.multiply(d3_error, sig(actvtn3, True))
            d1_error = back_prop(actvtn2, thetax, d2_error)

            dx_delta += numpy.dot(d2_error, actvtn2.T)
            d1_delta += numpy.dot(d1_error, actvtn1.T)

            theta1 -= alpha * (d1_delta / samples + (lamb * theta1))
            thetax -= alpha * (dx_delta / samples + (lamb * thetax))

        elif mode == 2: # stochastic gradient descent
            # Update for each sample
            for j in range(samples):
                # Setup input / output data
                in_mat = myx[:, j]
                out_mat = myy[:, j]

                # Forward Propogation
                actvtn1 = in_mat
                actvtn2 = fwd_prop(actvtn1, theta1)
                actvtn3 = fwd_prop(actvtn2, thetax)

                # Back Propogation
                d3_error = actvtn3 - out_mat

                calc_err(i, j, d3_error)

                # d2_error = back_prop(actvtn2, thetax, d3_error)
                # d1_error = back_prop(actvtn1, theta1, d2_error)

                # dx_delta += numpy.dot(d3_error, actvtn2.T)
                # d1_delta += numpy.dot(d2_error, actvtn1.T)

                d2_error = numpy.multiply(d3_error, sig(actvtn3, True))
                d1_error = back_prop(actvtn2, thetax, d2_error)

                dx_delta += numpy.dot(d2_error, actvtn2.T)
                d1_delta += numpy.dot(d1_error, actvtn1.T)

                theta1 -= alpha * (d1_delta / samples + (lamb * theta1))
                thetax -= alpha * (dx_delta / samples + (lamb * thetax))
        elif mode == 3: # mini-batch gradient descent
            # Process 'mini-batch' number of samples, then update
            for j in range(samples):
                # Setup input / output data
                in_mat = myx[:, j]
                out_mat = myy[:, j]

                # Forward Propogation
                actvtn1 = in_mat
                actvtn2 = fwd_prop(actvtn1, theta1)
                actvtn3 = fwd_prop(actvtn2, thetax)

                # Back Propogation
                d3_error = actvtn3 - out_mat

                calc_err(i, j, d3_error)

                # d2_error = back_prop(actvtn2, thetax, d3_error)
                # d1_error = back_prop(actvtn1, theta1, d2_error)

                # dx_delta += numpy.dot(d3_error, actvtn2.T)
                # d1_delta += numpy.dot(d2_error, actvtn1.T)

                d2_error = numpy.multiply(d3_error, sig(actvtn3, True))
                d1_error = back_prop(actvtn2, thetax, d2_error)

                dx_delta += numpy.dot(d2_error, actvtn2.T)
                d1_delta += numpy.dot(d1_error, actvtn1.T)

                if (j % batchsize == 0) & (j != 0):
                    theta1 -= alpha * (d1_delta / samples + (lamb * theta1))
                    thetax -= alpha * (dx_delta / samples + (lamb * thetax))

            # Update one final time to account for the remaining samples
            theta1 -= alpha * (d1_delta / samples + (lamb * theta1))
            thetax -= alpha * (dx_delta / samples + (lamb * thetax))

    # Check the results by analyzing one training example (in this case 0)
    check1 = myx
    check2 = calc_z(check1, theta1)
    check3 = sig(check2)
    check3 = calc_z(check3, thetax)
    check4 = sig(check3)
    print(numpy.round(100 * check4))

    return theta1, thetax
