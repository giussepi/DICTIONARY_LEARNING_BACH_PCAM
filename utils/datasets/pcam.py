# -*- coding: utf-8 -*-
"""
utils/datasets/pycam

Classes and methods to read and process data from PatchCamelyon (PCam)

https://github.com/basveeling/pcam
"""

import csv
import os
from collections import defaultdict

import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

import settings
from constants.constants import PCamLabel
from utils.utils import clean_create_folder


class HDF5_2_PNG:
    """
    Converts the HDF5 files from PathCamelyon dataset into PNG images and create a
    structure that can be consumed by the application

    Usage:
        HDF5_2_PNG()()
    """
    # Example PCam h5 filename": camelyonpatch_level_2_split_test_x.h5
    BASE_FILE_PATTERN = 'camelyonpatch_level_2_split_{}_{}.h5'
    SUB_DATASETS = ['train', 'valid', 'test']

    def __init__(self):
        """ Initialized the instance """
        self.base_folder_path = os.path.join(settings.BASE_DATASET_LINK, 'images')
        self.tumor_folder_path = os.path.join(self.base_folder_path, PCamLabel.TUMOR.name)
        self.normal_folder_path = os.path.join(self.base_folder_path, PCamLabel.NORMAL.name)

    def __call__(self):
        """ Functor call """
        self.__process()

    def __write_list_to_csv(self, list_, filename):
        """
        Writes the list into CSV format

        Args:
            list_   (list): List to be saved
            filename (str): CSV file name
        """
        assert isinstance(list_, list)
        assert len(list_) >= 1
        assert isinstance(filename, str)

        full_path = os.path.join(self.base_folder_path, filename)

        with open(full_path, 'w') as file_:
            writer = csv.writer(file_)

            if isinstance(list_[0], list):
                writer.writerows(list_)
            else:
                writer.writerow(list_)

    def __process(self):
        """ Process PatchCamelyon dataset and creates PNG images plus CSV files """
        normal_counter = 0
        tumor_counter = 0
        ground_truth = defaultdict(list)

        clean_create_folder(self.tumor_folder_path)
        clean_create_folder(self.normal_folder_path)

        for sub_dataset in self.SUB_DATASETS:
            y_path = os.path.join(
                settings.BASE_DATASET_LINK, self.BASE_FILE_PATTERN.format(sub_dataset, 'y'))
            x_path = os.path.join(
                settings.BASE_DATASET_LINK, self.BASE_FILE_PATTERN.format(sub_dataset, 'x'))

            with h5py.File(y_path, 'r') as y_file:
                with h5py.File(x_path, 'r') as x_file:
                    x_data = x_file['x']
                    y_data = y_file['y']

                    print("Creating PNG and CSV files for {} sub-dataset".format(sub_dataset))
                    for i in tqdm(range(y_data.shape[0])):
                        if y_data[i][0][0][0] == PCamLabel.TUMOR.id:
                            tumor_counter += 1
                            filename = 't{}.png'.format(tumor_counter)
                            saving_path = os.path.join(self.tumor_folder_path, filename)
                            label = PCamLabel.TUMOR.name
                        else:
                            normal_counter += 1
                            filename = 'n{}.png'.format(normal_counter)
                            saving_path = os.path.join(self.normal_folder_path, filename)
                            label = PCamLabel.NORMAL.name

                        plt.imsave(saving_path, x_data[i], format='png')
                        ground_truth[sub_dataset].append([filename, label])

                    self.__write_list_to_csv(
                        ground_truth[sub_dataset], '{}.csv'.format(sub_dataset))

        full_ground_truth = []
        print("Creating full ground truth CSV file")
        with tqdm(total=2) as pbar:
            for values in ground_truth.values():
                full_ground_truth.extend(values)
            pbar.update(1)
            self.__write_list_to_csv(full_ground_truth, 'full_ground_truth.csv')
            pbar.update(1)
