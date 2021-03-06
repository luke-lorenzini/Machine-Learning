def logreg(x, y, t, a, s):
    import math
    import numpy as np

    myx_o = x
    myy_o = y

    myx = myx_o / myx_o.max(axis=0)
    myy = myy_o / myy_o.max(axis=0)

    theta = t
    alpha = a
    threshold = 0.000001

    numrows = myx.shape[0]
    numcols = myx.shape[1]
    theta_t = np.matrix(np.zeros(numcols))
    # Make the difference large
    difference = 3 * theta

    #h_theta(x) = 1/(1+e^(-theta^T * x))
    def calc_h(row):
        h_theta = 0
        for i in range(numcols):
            h_theta += theta[0, i] * myx[row, i]
        xx = 1 / (1 + math.exp(-1 * h_theta))
        return xx
        # return h_theta

    def v_calc_h():
        h_theta = myx * theta.T
        xx = 1 / (1 + np.exp(-1 * h_theta))
        return xx

    learn = alpha / numrows
    # while abs(difference[0, 0]) > threshold:
    for repeats in range(10000):
        if s == 0: # Slowest - Basic implementation, no optimizations
            for col in range(numcols):
                temp = 0
                for row in range(numrows):
                    temp += (calc_h(row) - myy[row, 0]) * myx[row, col]
                theta_t[0, col] = theta[0, col] - alpha / numrows * temp
        elif s == 1: # Slower - Uses vectorized calculation for theta to speed things up
            temp_theta = v_calc_h()
            for col in range(numcols):
                temp = 0
                for row in range(numrows):
                    temp += (temp_theta[row, 0] - myy[row, 0]) * myx[row, col]
                theta_t[0, col] = theta[0, col] - alpha / numrows * temp
        elif s == 2: # Slow (the fastest)
            # theta_t = theta - (alpha / m) * X.T(g(X*theta) - Y)
            temp = theta.T - learn * myx.T * (v_calc_h() - myy)
            theta_t = temp.T

        # Comparison to determine stop condition
        difference = theta - theta_t

        # Update the values for theta
        np.copyto(theta, theta_t)
        # print(theta)
    
    check = myx * theta.T
    result = 1 / (1 + np.exp(-1 * check))
    print(result)

    return theta
