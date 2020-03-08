# -*- coding: utf-8 -*-
"""  """

import h5py
import matplotlib.pyplot as plt
import numpy as np


def main():
    """  """
    ###########################################################################
    #                            Reading hdf5 files                           #
    ###########################################################################
    f = h5py.File('datasets/PatchCamelyon (PCam)/camelyonpatch_level_2_split_test_y.h5', 'r')
    labels = np.array(f['y']).ravel()
    f.close()
    with h5py.File('datasets/PatchCamelyon (PCam)/camelyonpatch_level_2_split_test_x.h5', 'r') as f:
        print(f['x'].shape)
        for i in range(0, 500, 20):
            plt.imshow(f['x'][i])
            plt.show()


if __name__ == '__main__':
    main()
