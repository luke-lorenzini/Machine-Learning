# Principal Component Analysis
def prince(x, rate):
    import numpy

    sigma = numpy.dot(x.T, x)
    examples = x.shape[0]
    sigma = sigma / examples

    U, s, V = numpy.linalg.svd(sigma)

    myRate = rate / 100

    # Sum the "s" matrix
    s_tot = numpy.sum(s)

    k_tot = 0
    k = 0
    result = 0
    while result < myRate:
        k_tot += numpy.sum(s[k])
        result = k_tot / s_tot
        k += 1

    # Get submatrix of size [mxi]
    U_res = U[:, :k]
    z = x * U_res
    print(k)

    return z
