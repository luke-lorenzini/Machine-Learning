def linreg(x, y, t, a, s):
    import numpy as np

    myx = x
    myy = y
    theta = t
    alpha = a
    threshold = 0.000001

    numrows = myx.shape[0]
    numcols = myx.shape[1]
    theta_t = np.matrix(np.zeros(numcols))
    # Make the difference large
    difference = 3 * theta

    #h_theta(x) = 1/(1-e^(theta^T * x))
    def calc_h(row):
        h_theta = 0
        for i in range(numcols):
            h_theta += theta[0, i] * myx[row, i]
        return h_theta

    def v_calc_h():
        h_theta = myx * theta.T
        return h_theta

    learn = alpha / numrows
    while abs(difference[0, 0]) > threshold:
    # for repeats in range(1):
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
        print(theta)

    return theta
