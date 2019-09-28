import os
import scipy.misc
import numpy as np


def get_data(pardir, max_size=None, debug=False):
    file_list = os.listdir(pardir)

    if max_size is None:
        max_size = len(file_list)

    x_data = list()
    y_data = list()

    for f in file_list[:max_size]:
        data_path = os.path.join(pardir, f)
        im = scipy.misc.imread(data_path, flatten=False, mode='RGB')
        ims = scipy.misc.imresize(im,(60, 160, 3))
        x_data.append(ims)
        num = f.split('.')[0]
        N = len(num)
        y = np.zeros((N, 10))
        for n, i in enumerate(num):
            y[n][int(i)] = 1

        y_data.append(y)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data


if __name__ == '__main__':
    (X, Y) = get_data()
    print(X.shape)
    print(Y.shape)
