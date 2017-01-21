import os
from PIL import Image
import numpy as np

file_set = os.listdir('data/')

for f in file_set:
    print(f)
    im = Image.open('data/{}'.format(f)).convert('1')
    pixels = list(im.getdata())
    width, height = im.size
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
    print(pixels)
    print(np.size(pixels))
    print(im.getdata())
    print(im.size)
    x_data = np.array(pixels)
    print(x_data)
    num = f.split('.')[0]
    y_data = np.zeros((5, 10))
    for n, i in enumerate(num):
        y_data[n][i] = 1

    print(num)
    print(y_data)
    im.save('output2.png')
