def linreg(x, y, t, a):
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
    lamb = 1000
    scale = 1

    def calc_h(row):
        myh = 0
        for i in range(numcols):
            myh += (theta[0, i] * myx[row, i])
        return myh

    def calc_j(col):
        myj = 0
        for row in range(numrows):
            myj += (calc_h(row) - myy[row]) *  myx[row, col]
        myj /= numrows
        return myj

    #scale = (1 - alpha * lamb / numrows)

    # theta_j = theta_j-alpha*d/dtheta(J_theta)
    while((abs(difference[0, 0]) > threshold) & (abs(difference[0, 1]) > threshold)):
        for col in range(numcols):
            if col == 0:
                theta_t[0, col] = theta[0, col] - alpha * calc_j(col)
            else:
                theta_t[0, col] = theta[0, col] * scale - alpha * calc_j(col)
        difference = theta - theta_t
        for col in range(numcols):
            theta[0, col] = theta_t[0, col]
        print(theta)

    return theta
