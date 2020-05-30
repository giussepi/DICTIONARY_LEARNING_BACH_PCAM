# -*- coding: utf-8 -*-
""" utils/fetature_descriptors/random_faces """

import numpy as np

import settings


class RandomFaces:
    """
    Calculates random-face feature descriptors

    IMPORTANT NOTE: Always use the same rand_matrix when iterating over the inputs images
    to extract the descriptors

    Usage:
        randface = RandomFaces(img_height=512, img_width=512)
        randface.get_rand_face_feature_descriptor(image)
    """

    def __init__(self, img_height, img_width, fd_dimension=settings.FD_DIMENSION, concat_chnls=False):
        """"
        Initializes the instance

        Args:
            img_height    (int): height of the images
            img_width     (int): width of images
            fd_dimension  (int): dimension of random-face feature descriptor
            concat_chnls (bool): If True the number of cols will be times three
        """
        assert isinstance(img_height, int)
        assert isinstance(img_width, int)
        assert isinstance(fd_dimension, int)
        assert isinstance(concat_chnls, bool)

        self.img_height = img_height
        self.img_width = img_width
        self.fd_dimension = fd_dimension
        self.cols_multiplier = 3 if concat_chnls else 1
        self.rand_matrix = self._generate_random_matrix()

    def _generate_random_matrix(self):
        """
        Generates the random matrix

        Returns:
            random matrix [self.fd_dimension, self.img_height * self.img_width * self.cols_multiplier] (numpy.narray)
        """
        rand_matrix = np.random.randn(
            self.fd_dimension, self.img_height * self.img_width * self.cols_multiplier)
        # TODO: review if on lcksvb I replaced eps by np.spacing(1)
        l2_norms = np.sqrt(np.sum(rand_matrix * rand_matrix, axis=1) + np.spacing(1))
        rand_matrix = rand_matrix/np.tile(l2_norms, (rand_matrix.shape[1], 1)).T

        return rand_matrix

    def get_feature_descriptor(self, image):
        """
        Returns the feature descriptor

        Args:
            image (np.ndarray)

        Returns:

            feature descriptor (np.ndarray) (self.fd_dimension, )
        """
        assert isinstance(image, np.ndarray)

        return self.rand_matrix @ image
