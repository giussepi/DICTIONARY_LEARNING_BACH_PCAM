# -*- coding: utf-8 -*-
""" settings """

import os

###############################################################################
#                            General Configuration                            #
###############################################################################

BASE_DATASET_LINK = os.path.join('datasets', 'ICIAR 2018 BACH challenge')

TRAIN_PHOTOS_DATASET = os.path.join(BASE_DATASET_LINK, 'ICIAR2018_BACH_Challenge', 'Photos')

TEST_PHOTOS_DATASET = os.path.join(BASE_DATASET_LINK, 'ICIAR2018_BACH_Challenge_TestDataset', 'Photos')


###############################################################################
#                         utils.datasets Configuration                        #
###############################################################################
# I can create all the minipatches and save the train, validation and test filenames or
# patterns into a pickle file to have reproductible results and also to not make the
# minipatches creattion more complicated
OUTPUT_FOLDER = 'output'
LABELS_FILENAME = 'labels.json'

HOLDBACK = 0.7
SMALLLIM = 0.3
CUT_SIZE = 512
OVERLAP_COEFFICIENT = 0.5
# overlap must be an integer to avoid errors with the sliding window algorithm
OVERLAP = int(OVERLAP_COEFFICIENT * CUT_SIZE)
