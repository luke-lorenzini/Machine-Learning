from ML.RNN import RNN
import numpy as np

WORDS = 128

FILENAME = r'..\Machine-Learning\Test-Data\Sample Text\sample.txt'

f = open(FILENAME, 'r')
x = f.read()

in_data = {}

# Our input data
# in_data = np.zeros((len(x), WORDS))

for i in range(len(x)):
    in_data[i] = np.zeros((WORDS, 1))
    myChar = ord(x[i])

    # # Convert into one hot
    in_data[i][myChar] = 1

o = RNN(in_data, WORDS)

print(o)
