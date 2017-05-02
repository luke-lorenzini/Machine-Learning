import numpy as np
from PIL import Image

im = Image.open(r"C:\Users\Luke Lorenzini\Pictures\sample pics\Luke-2017-03-31_105006.bmp")
p = np.array(im)

print(p)
