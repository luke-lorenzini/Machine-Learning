def one_hot_encode(onehot, N):
    import numpy as np

    # Convert from integer to one hot, crude, but functional
    # takes an [mx1] and converts to [mxN]
    # newy = np.matrix(np.zeros((onehot.shape[0], N)))
    newyy = np.matrix(np.zeros((onehot.shape[0], N)))
    for i in range(onehot.shape[0]):
        for digit in range(N):
            if (onehot[i, 0] == digit):
                newyy[i, digit] = 1
            else:
                newyy[i, digit] = 0
        # if (onehot[i, 0] == 0):
        #     newy[i] = np.matrix([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # elif (onehot[i, 0] == 1):
        #     newy[i] = np.matrix([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        # elif (onehot[i, 0] == 2):
        #     newy[i] = np.matrix([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        # elif (onehot[i, 0] == 3):
        #     newy[i] = np.matrix([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        # elif (onehot[i, 0] == 4):
        #     newy[i] = np.matrix([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        # elif (onehot[i, 0] == 5):
        #     newy[i] = np.matrix([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        # elif (onehot[i, 0] == 6):
        #     newy[i] = np.matrix([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        # elif (onehot[i, 0] == 7):
        #     newy[i] = np.matrix([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        # elif (onehot[i, 0] == 8):
        #     newy[i] = np.matrix([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        # elif (onehot[i, 0] == 9):
        #     newy[i] = np.matrix([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    return (newyy)
