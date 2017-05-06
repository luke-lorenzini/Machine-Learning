def RNN():
    import numpy as np
    from ML.helpers import softmax
    from ML.helpers import sig
    from ML.helpers import tanh

    hidden = 100
    words = 8000
    steps = 50

    np.arange(1)

    # x_t = words x 1
    # o_t = words x 1
    # s_t = hidden x 1
    # U = hidden x words
    # V = words x hidden
    # W = hidden x hidden
    x_t, o_t, s_t = {}, {}, {}

    for i in range(steps + 1):
        x_t[i] = np.zeros((words, 1))
        o_t[i] = np.zeros((words, 1))
        s_t[i] = np.zeros((hidden, 1))

    # For Vanilla
    U = np.zeros((hidden, words))
    W = np.zeros((hidden, hidden))

    # For GRU
    Uz = np.zeros((hidden, words))
    Wz = np.zeros((hidden, hidden))
    Ur = np.zeros((hidden, words))
    Wr = np.zeros((hidden, hidden))
    Uh = np.zeros((hidden, words))
    Wh = np.zeros((hidden, hidden))

    # For LSTM
    Ui = np.zeros((hidden, words))
    Wi = np.zeros((hidden, hidden))
    Uf = np.zeros((hidden, words))
    Wf = np.zeros((hidden, hidden))
    Uo = np.zeros((hidden, words))
    Wo = np.zeros((hidden, hidden))
    Ug = np.zeros((hidden, words))
    Wg = np.zeros((hidden, hidden))

    V = np.zeros((words, hidden))

    dldU = np.zeros(Uz.shape)

    def Vanilla(x_t1, s_t0):
        # Vanilla
        # s_t = tanh(U x_t + W s_(t-1))
        # o_t = softmax(V s_t)
        s_t1 = tanh(np.dot(U, x_t1) + np.dot(W, s_t0))
        o_t1 = softmax(np.dot(V, s_t1))
        return o_t1, s_t1

    def GRU(x_t1, s_t0):
        # GRU
        # z = sig(x_t U^z + s_(t-1) W^z)    # update
        # r = sig(x_t U^r + s_(t-1) W^r)    # reset
        # h = tanh(x_t U^h + (s_(t-1) ELEMENT_WISE_MULT r) W^h) # hidden
        # s_t = (1 - z) ELEMENT_WISE_MULT h + z ELEMENT_WISE_MULT s_(t-1)
        # o_t = softmax(V s_t)
        z = sig(np.dot(Uz, x_t1) + np.dot(Wz, s_t0))
        r = sig(np.dot(Ur, x_t1) + np.dot(Wr, s_t0))
        h = tanh(np.dot(Uh, x_t1) + np.dot(Wh, np.multiply(s_t0, r)))
        s_t1 = np.multiply((1 - z), h) + np.multiply(z, s_t0)
        o_t1 = softmax(np.dot(V, s_t1))
        return o_t1, s_t1

    def LSTM(x_t1, s_t0, c_t0):
        # LSTM
        # g = tanh(x_t U^g + s_(t-1) W^g)   # candidate
        # i = sig(x_t U^i + s_(t-1) W^i)    # input
        # f = sig(x_t U^f + s_(t-1) W^f)    # forget
        # o = sig(x_t U^o + s_(t-1) W^o)    # output
        # c_t = c_(t-1) ELEMENT_WISE_MULT f + g ELEMENT_WISE_MULT i # internal memory
        # s_t = tanh(c_t) ELEMENT_WISE_MULT o
        g = tanh(np.dot(Ug, x_t1) + np.dot(Wg, s_t0))
        i = sig(np.dot(Ui, x_t1) + np.dot(Wi, s_t0))
        f = sig(np.dot(Uf, x_t1) + np.dot(Wf, s_t0))
        o = sig(np.dot(Uo, x_t1) + np.dot(Wo, s_t0))
        c_t1 = np.multiply(c_t0, f) + np.multiply(g, i)
        s_t1 = np.multiply(tanh(c_t1), o)
        o_t1 = softmax(np.dot(V, s_t1))
        return o_t1, s_t1, c_t1

    for i in range(steps):
        o_t[i+1], s_t[i+1] = Vanilla(x_t[i+1], s_t[i])

    for i in range(steps):
        o_t[i+1], s_t[i+1] = GRU(x_t[i+1], s_t[i])

    for i in range(steps):
        o_t[i+1], s_t[i+1] = LSTM(x_t[i+1], s_t[i])

    return o_t, s_t
