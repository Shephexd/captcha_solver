import os
import scipy.misc
import numpy as np


def get_data(debug=False):
    file_set = os.listdir('data/')

    x_data = list()
    y_data = list()

    for f in file_set:
        im = scipy.misc.imread('data/{}'.format(f), flatten=False, mode='RGB')
        ims = scipy.misc.imresize(im,(60, 160, 3))
        x_data.append(ims)
        num = f.split('.')[0]
        N = len(num)
        y = np.zeros((N, 10))
        for n, i in enumerate(num):
            y[n][int(i)] = 1

        y_data.append(y)

    if debug:
        print(x_data.shape)
        print(y_data.shape)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # return numpy array
    return x_data, y_data

if __name__ == '__main__':
    (X, Y) = get_data()
    print(X.shape)
    print(Y.shape)
