# The normal equation
# (X^T*X)^-1*X^T*Y

def norm(x, y):
    import numpy as np
    from numpy.linalg import inv

    myx = x
    myy = y

    theta = inv(myx.T * myx) * myx.T * myy

    return theta
