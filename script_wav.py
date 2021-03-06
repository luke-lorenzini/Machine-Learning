# A Script to test concepts
from time import time
import numpy as np
from ML.PCA import prince
from ML.mean_normalization import meannorm
# from NNet import NeuralNet
from ML.neural_network import NeuralNet
from scipy import misc
import wave
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import math

# Read a wav file, convert it to L & R channels
filename = r'C:\Users\Luke Lorenzini\OneDrive\Visual Studio\Python\Machine-Learning\Test-Data\sound\wavTones.com.unregistred.sin_1000Hz_-6dBFS_3s.wav'
# filename = r'Machine-Learning\Test Data\sound\folk.wav'
ff = wave.open(filename, 'rb')
print(ff.getparams())
valuesx = np.zeros((ff.getnframes(), ff.getnchannels()))
valuesx = np.matrix(np.zeros((ff.getnframes(), ff.getnchannels())))
for x in range(ff.getnframes()):
    zz = ff.readframes(1)
    # 24 bit, 2 ch
    if (ff.getsampwidth() == 3) & (ff.getnchannels() == 2):
        temp0 = (zz[2] << 16) + (zz[1] << 8) + (zz[0])
        temp1 = (zz[5] << 16) + (zz[4] << 8) + (zz[3])
        if temp0 > 0x7fffff:
            temp0 -= 0xffffff
        if temp1 > 0x7fffff:
            temp1 -= 0xffffff

        valuesx[x, 0] = temp0
        valuesx[x, 1] = temp1
    # 16 bit, 1 ch
    elif (ff.getsampwidth() == 2) & (ff.getnchannels() == 1):
        temp0 = (zz[1] << 8) + (zz[0])
        # if temp0 > 0x7fff:
        #     temp0 -= 2**16
        valuesx[x] = np.int16(temp0)

# One period = periods * sample rate / frequency, sample file, 1kHz --> 2 * 44,100 / 1000 
# plt.plot(valuesx)
# plt.show()

n = len(valuesx)
p = np.fft.fft(valuesx[:, 0])

uniquePts = int(math.ceil((n+1)/2.0))
p = p[0:uniquePts]
p = abs(p)

p = p / float(n)
p = p**2

if n % 2 > 0:
    p[1:len(p)] = p[1:len(p)] * 2
else:
    p[1:len(p) - 1] = p[1:len(p) - 1] * 2

sampFreq = ff.getframerate()
freqArray = np.arange(0, uniquePts, 1.0) * (sampFreq / n)

plt.plot(freqArray / 1000, 10 * np.log10(p))
plt.show()

valuesx_f = np.fft.fft(valuesx[:, 0])
d = len(valuesx_f)/2
plt.plot(abs(valuesx_f[:(d-1)]), 'r')
plt.show()

valuesy = valuesx[1:valuesx.shape[0], :]
valuesx = valuesx[0:valuesx.shape[0] - 1, :]

# Use these when only viewing one column from dataset
if valuesx.shape[0] == 1:
    valuesx = valuesx.T
if valuesy.shape[0] == 1:
    valuesy = valuesy.T

# valuesz = valuesx

# valuesz = prince(valuesz)
valuesz = meannorm(valuesx)
valuesy = meannorm(valuesy)

timenow = time()
t1, t2 = NeuralNet(valuesz, valuesy)
print(time() - timenow)

np.savetxt("theta1.csv", t1, delimiter=",")
np.savetxt("theta2.csv", t2, delimiter=",")
