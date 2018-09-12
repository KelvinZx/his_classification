import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 10, 20


def show_dataset(dataset, n=6):
    """
    Dataset is the return from irondataset.
    :param dataset:
    :param n:
    :return:
    """
    img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(n)))
                   for i in range(len(dataset))))
    plt.imshow(img)
    plt.axis('off')