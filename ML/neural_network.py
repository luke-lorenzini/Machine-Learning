def NeuralNet(x, y):
    import numpy
    import numpy.matlib
    from ML.helpers import init_weights
    from ML.helpers import append
    from ML.helpers import calc_alpha
    from ML.helpers import calc_z
    from ML.helpers import sigmoid
    from ML.helpers import sigmoid_prime

    # mode = 0, batch gradient descent
    # mode = 1, vectorized batch gradient descent
    # mode = 2, stochastic gradient descent
    # mode = 3, mini-batch gradient descent
    mode = 3

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
    # numpy.random.seed(1)

    myx = x.T
    myy = y.T

    epochs = 1
    samples = myx.shape[1]
    lamb = 0.001
    # 0 = no bias on layers, 1 = one bias layer
    bias = 0
    batchsize = 100

    error = 1

    # arch represents network architecture (w/o biases)
    arch = (myx.shape[0], 2 * myx.shape[0], 2 * myy.shape[0], myy.shape[0])

    ### Initialization ###
    # Use an [mxn] matrix to create an [nx(n+1)]
    theta1 = init_weights(arch[1], arch[0] + bias)
    thetax = init_weights(arch[len(arch) - 1], arch[1] + bias)

    d1_delta = numpy.zeros((theta1.shape[0], theta1.shape[1]))
    dx_delta = numpy.zeros((thetax.shape[0], thetax.shape[1]))

    def fwd_prop(mat, theta):
        """Forward Propogation"""
        # Each column represents a training example

        # Step 2, calculate z
        z = calc_z(mat, theta)

        # Step 3, transform z
        act = sigmoid(z)

        return act

    def back_propn(j, i, mat, act1, act2, act3):
        """ Back Propogation"""
        # Calculate errors for layer 3
        # d3_error = act3 - myy[:, j]
        d3_error = act3 - mat

        if (i % (epochs / 10)) == 0:
            if j == 0:
                error = numpy.mean(numpy.abs(d3_error))
                print(100 * numpy.mean(numpy.abs(d3_error)))

        d2_error = numpy.multiply(numpy.dot(thetax.T, d3_error), sigmoid_prime(act2))

        # Calculate errors for layer 2
        if bias > 0:
            d2_error = numpy.delete(d2_error, 0, 0)
        d1_error = numpy.multiply(numpy.dot(theta1.T, d2_error), sigmoid_prime(act1))

        return d1_error, d2_error, d3_error

    def back_prop(j, i, mat, act1, act2, act3):
        """ Back Propogation"""
        # Calculate errors for layer 3
        # d3_error = act3 - myy[:, j]
        d3_error = act3 - mat

        if (i % (epochs / 10)) == 0:
            if j == 0:
                error = numpy.mean(numpy.abs(d3_error))
                print(100 * numpy.mean(numpy.abs(d3_error)))

        d2_error = numpy.multiply(numpy.dot(thetax.T, d3_error), sigmoid_prime(act2))

        # Calculate errors for layer 2
        if bias > 0:
            d2_error = numpy.delete(d2_error, 0, 0)
        d1_error = numpy.multiply(numpy.dot(theta1.T, d2_error), sigmoid_prime(act1))

        return d1_error, d2_error, d3_error

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
                if bias > 0:
                    actvtn1 = append(actvtn1)

                actvtn2 = fwd_prop(actvtn1, theta1)
                if bias > 0:
                    actvtn2 = append(actvtn2)

                actvtn3 = fwd_prop(actvtn2, thetax)

                # Back Propogation
                d1_error, d2_error, d3_error = back_prop(j, i, out_mat, actvtn1, actvtn2, actvtn3)

                dx_delta += numpy.dot(d3_error, actvtn2.T)
                d1_delta += numpy.dot(d2_error, actvtn1.T)

            theta1 -= alpha * (d1_delta / samples + (lamb * theta1))
            thetax -= alpha * (dx_delta / samples + (lamb * thetax))

        elif mode == 1: # vectorized batch gradient descent
            # Calculate the whole shebang
            # Setup input / output data
            in_mat = myx
            out_mat = myy

            # Forward Propogation
            actvtn1 = in_mat
            if bias > 0:
                actvtn1 = append(actvtn1)

            actvtn2 = fwd_prop(actvtn1, theta1)
            if bias > 0:
                actvtn2 = append(actvtn2)

            actvtn3 = fwd_prop(actvtn2, thetax)

            # Back Propogation
            d1_error, d2_error, d3_error = back_prop(0, i, out_mat, actvtn1, actvtn2, actvtn3)

            dx_delta += numpy.dot(d3_error, actvtn2.T)
            d1_delta += numpy.dot(d2_error, actvtn1.T)

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
                if bias > 0:
                    actvtn1 = append(actvtn1)

                actvtn2 = fwd_prop(actvtn1, theta1)
                if bias > 0:
                    actvtn2 = append(actvtn2)

                actvtn3 = fwd_prop(actvtn2, thetax)

                # Back Propogation
                d1_error, d2_error, d3_error = back_prop(j, i, out_mat, actvtn1, actvtn2, actvtn3)

                dx_delta += numpy.dot(d3_error, actvtn2.T)
                d1_delta += numpy.dot(d2_error, actvtn1.T)

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
                if bias > 0:
                    actvtn1 = append(actvtn1)

                actvtn2 = fwd_prop(actvtn1, theta1)
                if bias > 0:
                    actvtn2 = append(actvtn2)

                actvtn3 = fwd_prop(actvtn2, thetax)

                # Back Propogation
                d1_error, d2_error, d3_error = back_prop(j, i, out_mat, actvtn1, actvtn2, actvtn3)

                dx_delta += numpy.dot(d3_error, actvtn2.T)
                d1_delta += numpy.dot(d2_error, actvtn1.T)

                if (j % batchsize == 0) & (j != 0):
                    theta1 -= alpha * (d1_delta / samples + (lamb * theta1))
                    thetax -= alpha * (dx_delta / samples + (lamb * thetax))

            # Update one final time to account for the remaining samples
            theta1 -= alpha * (d1_delta / samples + (lamb * theta1))
            thetax -= alpha * (dx_delta / samples + (lamb * thetax))

    # Check the results by analyzing one training example (in this case 0)
    check1 = myx
    if bias > 0:
        check1 = append(check1)
    check2 = calc_z(check1, theta1)
    check3 = sigmoid(check2)
    if bias > 0:
        check3 = append(check3)
    check3 = calc_z(check3, thetax)
    check4 = sigmoid(check3)
    print(numpy.round(100 * check4))

    return theta1, thetax
