# -*- coding: utf-8 -*-
"""
utils/datasets/pycam

Classes and methods to read and process data from PatchCamelyon (PCam)

https://github.com/basveeling/pcam
"""

import csv
import json
import os
from collections import defaultdict

import h5py
import numpy as np
import torch
from gtorch_utils.datasets.generic import BaseDataset
from gutils.numpy_.numpy_ import LabelMatrixManager
from PIL import Image
from tqdm import tqdm

import settings
from constants.constants import PCamLabel, PCamSubDataset
from core.exceptions.dataset import PCamImageNameInvalid
from utils.datasets.bach import BasePrepareDataset as BachBasePrepareDataset
from utils.datasets.mixins import CreateJSONFilesMixin
from utils.utils import clean_create_folder, load_codes, clean_json_filename, \
    get_filename_and_extension


class HDF5_2_PNG(PCamSubDataset):
    """
    Converts the HDF5 files from PathCamelyon dataset into PNG images and create a
    structure that can be consumed by the application

    Usage:
        HDF5_2_PNG(only_center=True)()
    """
    # Example PCam h5 filename": camelyonpatch_level_2_split_test_x.h5
    BASE_FILE_PATTERN = 'camelyonpatch_level_2_split_{}_{}.h5'

    def __init__(self, only_center=False):
        """
        Initialized the instance

        Args:
            only_center (bool): If true the each image will include only the 32x32 pixels centre
                                 used during labelling (PatchCamelyon dataset feature).
        """
        self.tumor_folder_path = os.path.join(settings.TRAIN_PHOTOS_DATASET, PCamLabel.TUMOR.name)
        self.normal_folder_path = os.path.join(settings.TRAIN_PHOTOS_DATASET, PCamLabel.NORMAL.name)
        self.only_center = only_center

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

        full_path = os.path.join(settings.TRAIN_PHOTOS_DATASET, filename)

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

                        if self.only_center:
                            plt_image = Image.fromarray((x_data[i][32:64, 32:64, :]).astype('uint8'))
                        else:
                            plt_image = Image.fromarray((x_data[i]).astype('uint8'))

                        plt_image.save(saving_path)
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


class FormatProvidedDatasetSplits(PCamSubDataset):
    """
    * Processes the train, valid and test datasets provided by PatchCamelyon and formatted
      by HDF5_2_PNG.
    * Creates CSV files in the settings.OUTPUT_FOLDER directory to be consumed by the application

    Usage:
        FormatProvidedDatasetSplits()()
    """

    def __init__(self):
        """ Initializes the instance """
        self.csv_output_filenames = [
            settings.TRAIN_SPLIT_FILENAME,
            settings.VALID_SPLIT_FILENAME,
            settings.TEST_SPLIT_FILENAME
        ]

    def __call__(self):
        """ Functor call """
        self.__process_splits()

    def __process_splits(self):
        """
        Processes the provided dataset splits and saves in CSV format to be consumed by the
        application
        """
        clean_create_folder(settings.OUTPUT_FOLDER)

        print("Formatting sub-datasets")
        for filename, sub_dataset in tqdm(list(zip(self.csv_output_filenames, self.SUB_DATASETS))):
            data = np.genfromtxt(
                os.path.join(settings.TRAIN_PHOTOS_DATASET, '{}.csv'.format(sub_dataset)),
                delimiter=',', dtype=np.str
            )
            formatted_csv_path = os.path.join(settings.OUTPUT_FOLDER, filename)
            with open(formatted_csv_path, 'w') as file_:
                json.dump(data.tolist(), file_)


class BasePrepareDataset(BachBasePrepareDataset):
    """
    Creates minipatches for each PNG file at settings.TRAIN_SPLIT_FILENAME,
    settings.VALID_SPLIT_FILENAME and settings.TEST_SPLIT_FILENAME; and saves them at
    settings.TRAIN_FOLDER_NAME, settings.VALID_FOLDER_NAME,
    settings.TEST_FOLDER_NAME folders respectively. Also creates the labels file
    for each train, valid and test folder.

    Usage:
        MiniPatch()()
    """

    def __init__(self, *args, **kwargs):
        """
        * Initializes the object
        * Cleans the train and ouput folders (performs delete and create operations)
        """
        super().__init__(*args, **kwargs)
        clean_create_folder(os.path.join(settings.OUTPUT_FOLDER, settings.VALID_FOLDER_NAME))
        self.split_files = [settings.TRAIN_SPLIT_FILENAME, settings.VALID_SPLIT_FILENAME,
                            settings.TEST_SPLIT_FILENAME]
        self.folder_names = [settings.TRAIN_FOLDER_NAME, settings.VALID_FOLDER_NAME,
                             settings.TEST_FOLDER_NAME]

    def _load_image_list(self):
        """
        Loads the train test json files, creates the paths to the TIFF images into a list, and
        assings the list to self.image_list.
        """
        super()._load_image_list()

        for filename, label in self.read_split_file(settings.VALID_SPLIT_FILENAME):
            self.image_list.append((
                os.path.join(settings.TRAIN_PHOTOS_DATASET, label, filename),
                settings.VALID_FOLDER_NAME
            ))

    def _create_labels(self):
        """
        Read the files from train and test folders and creates a dictionary with
        the format key: filename, value: label. Each dictionary is saved as a JSON
        file in their correspoding folder with the name settings.LABELS_FILENAME.
        """
        for folder in self.folder_names:
            folder_path = os.path.join(settings.OUTPUT_FOLDER, folder)
            filenames = os.listdir(folder_path)
            labels = []
            print('Creating labels file for {} dataset'.format(folder))

            for filename in tqdm(filenames):
                if filename.startswith('t'):
                    labels.append(PCamLabel.TUMOR.id)
                elif filename.startswith('n'):
                    labels.append(PCamLabel.NORMAL.id)
                else:
                    raise PCamImageNameInvalid()

            file_path = os.path.join(folder_path, settings.LABELS_FILENAME)

            with open(file_path, 'w') as file_:
                json.dump(dict(zip(filenames, labels)), file_)


class WholeImage(CreateJSONFilesMixin, BasePrepareDataset):
    """
    Creates JSON files covering the whole PNG images from settings.TRAIN_SPLIT_FILENAME,
    settings.VALID_SPLIT_FILENAME and settings.TEST_SPLIT_FILENAME files; and saves them
    at settings.TRAIN_FOLDER_NAME, settings.VALID_FOLDER_NAME and
    settings.TEST_FOLDER_NAME folders respectively. Also for creates the labels file
    for train, valid and test folders.

    Use it to work with the whole images instead of patches.

    Usage:
        WholeImage()()
    """


class PCamTorchDataset(BaseDataset):
    """  """

    def __init__(self, subset, **kwargs):
        """
        Loads the subdataset

        Args:
           filename_pattern (str): filename with .json extension used to create the codes
                                   when the calling the create_datasets_for_LC_KSVD method.
           code_type   (CodeType): Code type used. See constants.constants.CodeType class defition
        """
        assert subset in PCamSubDataset.SUB_DATASETS
        self.subset = subset
        filename_pattern = kwargs.get('filename_pattern')
        assert isinstance(filename_pattern, str)

        code_type = kwargs.get('code_type')
        cleaned_filename = clean_json_filename(filename_pattern)
        name, extension = get_filename_and_extension(cleaned_filename)
        file_name = '{}_{}.{}'.format(name, subset, extension)
        self.data = load_codes(file_name, type_=code_type)
        self.data['labels'] = LabelMatrixManager.get_1d_array_from_2d_matrix(self.data['labels'])

    def __len__(self):
        """
        Returns:
            dataset size (int)
        """
        return self.data['labels'].shape[0]

    def __getitem__(self, idx):
        """
        Returns:
            dict(feats=..., label=...)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return dict(
            feats=torch.from_numpy(self.data['codes'][:, idx].ravel()).float(),
            label=self.data['labels'][idx]
        )
