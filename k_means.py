def kmeans(x, K):
    import math
    import numpy
    import numpy.matlib
    import sys

    myx = x
    # u = numpy.zeros(K)
    samples = myx.shape[0]
    c = numpy.zeros(samples)

    cent = numpy.matlib.rand(K, myx.shape[1])
    # cent = 10 * numpy.random.random((K - 1, myx.shape[1])) - 1

    for total in range(10):
        hits = numpy.zeros(K)

        # Cluster assignment
        # c[i] contains information about the cluster which example belongs
        for	i in range(samples):
            # Set to something big
            zzz = sys.maxsize
            # Calculate scalar between sample[i] and centroid[k]
            for k in range(K):
                # sqrt((x2-x1)^2 + (y2-y1)^2)
                mysum = 0
                for cols in range(myx.shape[1]):
                    mysum += (myx[i, cols] - cent[k, cols]) ** 2
                dist = math.sqrt(mysum)
                # print(dist)
                if dist < zzz:
                    # Flag the smallest distance for comparison
                    zzz = dist
                    x = k
            for k in range(K):
                if x == k:
                    hits[k] += 1
            # Calculate distance from point i
            # c_i :=	index	 (from	1	to	K)	of	cluster	 centroid	 closest	to	x_i
            c[i] = x

        # Move centroid
        for k in range(K):
            # u_k :=	average	(mean)	of	points	assigned	to	 cluster	k
            sums = numpy.zeros(myx.shape[1])

            for	i in range(samples):
                if k == c[i]:
                    for cols in range(myx.shape[1]):
                        sums[cols] += myx[i, cols]
            if hits[k] != 0:
                sums /= hits[k]
            # Update the centroid matrix
            cent[k, ] = sums

        # print(cent)
    print(c)
