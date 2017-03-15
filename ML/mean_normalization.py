# Mean Normalization
def meannorm(x):
    import numpy

    a = x
    b = numpy.average(a, axis=0)
    c = numpy.std(a, axis=0)

    # myx = x / x.max(axis=0)
    myx = (a - b) / (c + 0.00000001)

    return myx
