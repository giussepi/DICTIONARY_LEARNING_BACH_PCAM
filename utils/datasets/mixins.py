# -*- coding: utf-8 -*-
""" utils/datasets/mixins """

import json
import os

import matplotlib.pyplot as plt
from tqdm import tqdm

import settings
from constants.constants import Label
from core.exceptions.dataset import ImageNameInvalid


class CreateJSONFilesMixin:
    """
    Mixins to be used with descendants of utils.datasets.bach.BasePrepareDataset class
    """

    def _create_json_files(self):
        """
        Reads the images from self.image_list and creates the
        the WSI JSON files
        """
        print("Processing images to create whole image JSON files")
        for image_path, folder in tqdm(self.image_list):
            image = plt.imread(image_path)
            h, w = image.shape[:2]

            self._create_image_json_file(
                self._format_clean_filename(image_path, 0, 0),
                folder, image_path, 0, 0, w, h
            )


class ReadSplitFileMixin:
    """ Mixin with a method to read a split file and returns its content """

    @staticmethod
    def read_split_file(filename, split_files):
        """
        Verifes the filename, loads the files and returns its content

        Args:
            filemame    (str): file to read
            split_files (str): list of split files

        Returns:
            content of split file
        """
        assert filename in split_files
        filepath = os.path.join(settings.OUTPUT_FOLDER, filename)
        assert os.path.isfile(filepath)

        with open(filepath, 'r') as file_:
            data = json.load(file_)

        return data


class CreateLabelsMixin:
    """ Mixin with a method to read files directories and create their label files """

    @staticmethod
    def _create_labels(folder_names):
        """
        Reads the files from train, validation and test folders and creates a dictionary with
        the format key: filename, value: label. Each dictionary is saved as a JSON
        file in their correspoding folder with the name settings.LABELS_FILENAME.

        Args:
            folder_names (str): list of forlder containing the files
        """
        for folder in folder_names:
            folder_path = os.path.join(settings.OUTPUT_FOLDER, folder)
            filenames = os.listdir(folder_path)
            labels = []
            print('Creating labels file for {} dataset'.format(folder))

            for filename in tqdm(filenames):
                if filename.startswith('b'):
                    labels.append(Label.BENIGN.id)
                elif filename.startswith('is'):
                    labels.append(Label.INSITU.id)
                elif filename.startswith('iv'):
                    labels.append(Label.INVASIVE.id)
                elif filename.startswith('n'):
                    labels.append(Label.NORMAL.id)
                else:
                    raise ImageNameInvalid()

            file_path = os.path.join(folder_path, settings.LABELS_FILENAME)

            with open(file_path, 'w') as file_:
                json.dump(dict(zip(filenames, labels)), file_)
