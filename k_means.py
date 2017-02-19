def kmeans(x, K):
    import math
    import numpy
    import numpy.matlib

    myx = x
    u = numpy.zeros(K)
    samples = myx.shape[0]
    c = numpy.zeros(samples)

    cent = numpy.matlib.rand(K, myx.shape[1])

    for z in range(10):
        zeros = 0
        ones = 0
        twos = 0
        threes = 0

        # Cluster assignment
        # c[i] contains information about the cluster which example belongs
        for	i in range(samples):
            # Set to something big
            zzz = 100000000000
            # Calculate scalar between sample[i] and centroid[k]
            for k in range(K):
                # sqrt((x2-x1)^2 + (y2-y1)^2)
                dist = math.sqrt((cent[k, 0] + myx[i, 0])**2 + (cent[k, 1] + myx[i, 1])**2 + (cent[k, 2] + myx[i, 2])**2 + (cent[k, 3] + myx[i, 3])**2)
                # print(dist)
                if dist < zzz:
                    # Note the smallest distance
                    zzz = dist
                    x = k
            if x == 0:
                zeros += 1
            elif x == 1:
                ones += 1
            elif x == 2:
                twos += 1
            elif x == 3:
                threes += 1
            # Calculate distance from point i
            # c_i :=	index	 (from	1	to	K)	of	cluster	 centroid	 closest	to	x_i
            c[i] = x

        # Move centroid
        for k in range(K):
            # u_k :=	average	(mean)	of	points	assigned	to	 cluster	k
            aaa = 0
            bbb = 0
            ccc = 0
            ddd = 0
            for	i in range(samples):
                if k == c[i]:
                    aaa += myx[i, 0]
                    bbb += myx[i, 1]
                    ccc += myx[i, 2]
                    ddd += myx[i, 3]
            if k == 0 & zeros != 0:
                aaa = aaa / zeros
                bbb = bbb / zeros
                ccc = ccc / zeros
                ddd = ddd / zeros
            elif k == 1 & ones != 0:
                aaa = aaa / ones
                bbb = bbb / ones
                ccc = ccc / ones
                ddd = ddd / ones
            elif k == 2 & twos != 0:
                aaa = aaa / twos
                bbb = bbb / twos
                ccc = ccc / twos
                ddd = ddd / twos
            elif k == 3 & threes != 0:
                aaa = aaa / threes
                bbb = bbb / threes
                ccc = ccc / threes
                ddd = ddd / threes
            # Update the centroid matrix
            cent[k, 0] = aaa
            cent[k, 1] = bbb
            cent[k, 2] = ccc
            cent[k, 3] = ddd

        print(cent)
