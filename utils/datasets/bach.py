# -*- coding: utf-8 -*-
"""
utils/datasets/bach

Classes and methods to read and process data from ICIAR 2018 BACH challenge

https://iciar2018-challenge.grand-challenge.org/Dataset/
"""

import json
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from constants.constants import Label
from core.exceptions.dataset import ImageNameInvalid
import settings
from utils.files import get_name_and_extension
from utils.utils import clean_create_folder


class MiniPatch:
    """
    Creates minipatches for each TIFF files at settings.TRAIN_SPLIT_FILENAME and
    settings.TEST_SPLIT_FILENAME and saves them at settings.TRAIN_FOLDER_NAME and
    settings.TRAIN_FOLDER_NAME folders respectively. Also for creates the labels file
    for each train and test folder.

    Usage:
        MiniPatch()()
    """

    def __init__(self, *args, **kwargs):
        """
        * Initializes the object
        * Cleans the train and ouput folders (performs delete and create operations)
        """
        self.image_list = None
        clean_create_folder(os.path.join(settings.OUTPUT_FOLDER, settings.TRAIN_FOLDER_NAME))
        clean_create_folder(os.path.join(settings.OUTPUT_FOLDER, settings.TEST_FOLDER_NAME))

    def __call__(self):
        """ Functor call """
        self.__load_image_list()
        self.__create_minipatches()
        self.__create_labels()

    @staticmethod
    def read_split_file(filename):
        """ Verifes the filename, loads the files and returns its content """
        assert filename in [settings.TRAIN_SPLIT_FILENAME, settings.TEST_SPLIT_FILENAME]
        filepath = os.path.join(settings.OUTPUT_FOLDER, filename)
        assert os.path.isfile(filepath)

        with open(filepath, 'r') as file_:
            data = json.load(file_)

        return data

    def __load_image_list(self):
        """
        Loads the train test json files, creates the paths to the TIFF images into a list, and
        assings the list to self.image_list.
        """
        self.image_list = []

        for filename, label in self.read_split_file(settings.TRAIN_SPLIT_FILENAME):
            self.image_list.append((
                os.path.join(settings.TRAIN_PHOTOS_DATASET, label, filename),
                settings.TRAIN_FOLDER_NAME
            ))

        for filename, label in self.read_split_file(settings.TEST_SPLIT_FILENAME):
            self.image_list.append((
                os.path.join(settings.TRAIN_PHOTOS_DATASET, label, filename),
                settings.TEST_FOLDER_NAME
            ))

    @staticmethod
    def __create_image_json_file(filename, folder, source_filename, x, y, xmax, ymax):
        """
        Creates a roi json file at settings.OUTPUT_FOLDER + folder + filename using the provided
        arguments

        Args:
            filename         (str): name of the file
            folder           (str): folder name
            source_filename  (str): path to the TIFF image
            x, y, xmax, ymax (int): region of interest (roi) coordinates
        """
        assert isinstance(filename, str)
        assert folder in [settings.TRAIN_FOLDER_NAME, settings.TEST_FOLDER_NAME]
        assert os.path.isfile(source_filename)
        assert isinstance(x, int)
        assert isinstance(y, int)
        assert isinstance(xmax, int)
        assert isinstance(ymax, int)

        with open(os.path.join(settings.OUTPUT_FOLDER, folder, filename), 'w') as _file:
            data = dict(
                source=source_filename,
                roi={'x': x, 'y': y, 'w': xmax - x, 'h': ymax - y}
            )
            json.dump(data, _file)

    @staticmethod
    def __format_clean_filename(filename, x_suffix, y_suffix):
        """
        Extracts and reformats the filename using the suffixes provided to create a file name
        for its json file

        Args:
            filename        (str): file name
            x_suffix (str or int): X
            y_suffix (str or int): y

        Returns:
           '<original_filename>_<x>_<y>.json'
        """
        name, _ = get_name_and_extension(filename.split('/')[-1])
        genfilename = "{}_{}_{}.json".format(name, x_suffix, y_suffix)

        return genfilename.replace(" ", "_")

    def __create_minipatches(self):
        """
        Reads the images from self.image_list and creates the
        minipatch json files
        """
        print("Processing TIFF images to create minipathes")
        for image_path, folder in tqdm(self.image_list):
            image = plt.imread(image_path)
            h, w = image.shape[:2]
            y = 0

            while y <= (h-settings.CUT_SIZE):
                x = 0
                while x <= (w-settings.CUT_SIZE):
                    self.__create_image_json_file(
                        self.__format_clean_filename(image_path, x, y), folder, image_path,
                        x, y,
                        x+settings.CUT_SIZE, y+settings.CUT_SIZE
                    )

                    x += settings.OVERLAP

                if (x-settings.CUT_SIZE) <= (settings.HOLDBACK*settings.CUT_SIZE):
                    x = w - settings.CUT_SIZE

                    self.__create_image_json_file(
                        self.__format_clean_filename(image_path, x, y), folder, image_path,
                        x, y,
                        w, y+settings.CUT_SIZE
                    )

                y += settings.OVERLAP

            if ((h/settings.CUT_SIZE) - (h//settings.CUT_SIZE)) >= settings.HOLDBACK:
                x = 0
                y = h - settings.CUT_SIZE
                while x <= (w-settings.CUT_SIZE):
                    self.__create_image_json_file(
                        self.__format_clean_filename(image_path, x, y), folder, image_path,
                        x, y,
                        x+settings.CUT_SIZE, y+settings.CUT_SIZE
                    )

                    x += settings.OVERLAP

                self.__create_image_json_file(
                    self.__format_clean_filename(image_path, w-settings.CUT_SIZE, h-settings.CUT_SIZE),
                    folder, image_path,
                    w-settings.CUT_SIZE, h-settings.CUT_SIZE,
                    w, h
                )

    @staticmethod
    def __create_labels():
        """
        Read the files from train and test folders and creates a dictionary with
        the format key: filename, value: label. Each dictionary is saved as a JSON
        file in their correspoding folder with the name settings.LABELS_FILENAME.

        """
        for folder in [settings.TRAIN_FOLDER_NAME, settings.TEST_FOLDER_NAME]:
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


class TrainTestSplit:
    """
    Splits the dataset into train and test and saves them in CSV files

    Args:
        test_size   (float): test dataset size in range [0, 1]

    Usage:
        TrainTestSplit(test_size=0.2)()
    """

    def __init__(self, *args, **kwargs):
        """ Initializes the instance """
        self.test_size = kwargs.get('test_size', settings.TEST_SIZE)
        assert isinstance(self.test_size, float)
        self.train_xy = self.test_xy = None

    def __call__(self):
        """
        * Functor call
        * Splits the dataset into train and test subsets
        * Saves train and test dataset into JSON files
        """
        self.__split_dataset()
        self.__create_json_files()

    def __split_dataset(self):
        """ Splits the dataset into train and test """
        print("Loading ground truth file...")
        with tqdm(total=1) as pbar:
            ground_truth = np.genfromtxt(
                settings.TRAIN_PHOTOS_GROUND_TRUTH, delimiter=',', dtype=np.str)
            pbar.update(1)

        print("Splitting dataset with test = {}".format(self.test_size))
        with tqdm(total=1) as pbar:
            x_train, x_test, y_train, y_test = train_test_split(
                ground_truth[:, 0], ground_truth[:, 1], test_size=self.test_size,
                random_state=settings.RANDOM_STATE, stratify=ground_truth[:, 1]
            )
            self.train_xy = np.hstack((
                np.expand_dims(x_train, axis=1), np.expand_dims(y_train, axis=1)))
            self.test_xy = np.hstack((
                np.expand_dims(x_test, axis=1), np.expand_dims(y_test, axis=1)))
            pbar.update(1)

    def __create_json_files(self):
        """ Saves train and test datasets into JSON files """
        print("Saving train/test datset into JSON files...")

        file_paths = [os.path.join(settings.OUTPUT_FOLDER, filename)
                      for filename in [settings.TRAIN_SPLIT_FILENAME, settings.TEST_SPLIT_FILENAME]]

        clean_create_folder(settings.OUTPUT_FOLDER)

        for file_path in tqdm(file_paths):
            data = self.train_xy if file_path.endswith(settings.TRAIN_SPLIT_FILENAME) else self.test_xy
            with open(file_path, 'w') as file_:
                # Workaround to save numpy array without errors
                json.dump(data.tolist(), file_)


class BACHDataset(Dataset):
    """ BACH Dataset """

    def __init__(
            self, json_images_folder, labels_filename=settings.LABELS_FILENAME, transform=None):
        """
        Note: the labels file must be inside the json_images_folder
        * Makes sure the json_images_folder and labels_filename exists
        * Loads the labels
        * Initialises the instance
        """
        assert os.path.isdir(json_images_folder)
        labels_path = os.path.join(json_images_folder, labels_filename)
        assert os.path.isfile(labels_path)

        with open(labels_path, 'r') as file_:
            self.data = pd.DataFrame(list(json.load(file_).items()), columns=["filename", "label"])

        self.root_dir = json_images_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = read_roi_image(img_name)
        target = np.array(self.data.iloc[idx, 1])
        sample = {'image': image, 'target': target}

        if self.transform:
            image = self.transform(sample['image'])

        return {'image': image, 'target': target}


def read_roi_image(file_path):
    """
    Reads the image from the roi_file and returns the ROI as a numpy array

    Args:
        file_path         (str): relative path to the iamge json file

    Returns:
        ROI numpy array
    """
    assert isinstance(file_path, str)
    assert os.path.isfile(file_path)

    with open(file_path, 'r') as file_:
        data = json.load(file_)
        image = plt.imread(data['source'])[
            data['roi']['y']:data['roi']['y']+data['roi']['h'],
            data['roi']['x']:data['roi']['x']+data['roi']['w'],
        ]

    return image


def plot_json_img(file_path, figsize=None, save_to_disk=False, folder_path='', carousel=False):
    """
    Reads a json image file and based on the provided parameters it can be plotted or
    saved to disk.

    Args:
        file_path         (str): relative path to the iamge json file
        figsize (None or tuple): dimensions of the image to be plotted
        save_to_disk     (bool): if true the image is saved to disk, otherwise it's plotted
        folder_path       (str): relative path to the folder where the image will be saved
        carousel         (bool): shows images consecutively only if it has been called through plot_n_first_json_images

    Usage:
        plot_json_img(os.path.join(settings.OUTPUT_FOLDER, 'b001_0_0.json'), (9, 9), False)
    """
    assert isinstance(file_path, str)
    assert os.path.isfile(file_path)
    assert isinstance(save_to_disk, bool)
    assert isinstance(folder_path, str)
    assert isinstance(carousel, bool)

    if folder_path and not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    if figsize:
        assert isinstance(figsize, tuple)
        assert len(figsize) == 2
        assert figsize[0] > 0 and figsize[1] > 0
    else:
        figsize = (8, 8)

    plt.figure(figsize=figsize)
    image = read_roi_image(file_path)
    plt.imshow(image)

    if save_to_disk:
        name, _ = get_name_and_extension(file_path.split('/')[-1])
        plt.savefig(os.path.join(folder_path, '{}.png'.format(name)))
    else:
        if carousel:
            plt.pause(1)
            plt.close()
        else:
            plt.show()


def plot_n_first_json_images(
        n_images, read_folder_path, figsize=None, save_to_disk=False, save_folder_path='',
        clean_folder=False, carousel=False):
    """
    Reads the n-fist json images from read_folder_path and based on the provided parameters
    they can be plotted or saved to disk.

    Args:
        n_images          (int): number of images to read
        read_folder_path  (str): folder containing the json images
        figsize (None or tuple): dimensions of the image to be plotted
        save_to_disk     (bool): if true the image is saved to disk, otherwise it's plotted
        save_folder_path (str): relative path to the folder where the image will be saved
        clean_folder     (bool): if true the folder is deleted and re-created
        carousel         (bool): shows images consecutively

    Usage:
        plot_n_first_json_images(5, os.path.join(settings.OUTPUT_FOLDER, settings.TRAIN_FOLDER_NAME),
                                (9, 9), False, 'my_folder', False, True)
    """
    assert isinstance(n_images, int)
    assert isinstance(clean_folder, bool)

    if clean_folder and os.path.isdir(save_folder_path):
        shutil.rmtree(save_folder_path)

    print("Plotting images")
    for image in tqdm(os.listdir(read_folder_path)[:n_images]):
        plot_json_img(
            os.path.join(read_folder_path, image), figsize, save_to_disk, save_folder_path, carousel)
