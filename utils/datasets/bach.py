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
# from PIL import Image
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
    Reads TIF files and labels from CSV file and creates
    minipatches using ROI json with their respective xml annotations

    Usage:
        MiniPatch()()
        MiniPatch(cut_size=608)()
    """

    def __init__(self, *args, **kwargs):
        """
        * Initializes the object
        * Clean the content of anno_location folder
        """
        self.path_image = kwargs.get('path_image', [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(settings.TRAIN_PHOTOS_DATASET)) for f in fn])
        self.path_anno = kwargs.get('path_anno', os.path.join(
            settings.TRAIN_PHOTOS_DATASET, 'microscopy_ground_truth.csv'))
        self.image_list = list(filter(lambda path: path.endswith('.tif'), self.path_image))
        self.anno_location = self.image_location = settings.OUTPUT_FOLDER
        self.holdback = kwargs.get('holdback', settings.HOLDBACK)
        self.smalllim = kwargs.get('smallim', settings.SMALLLIM)
        self.cut_size = kwargs.get('cut_size', settings.CUT_SIZE)
        self.overlap_coefficient = kwargs.get('overlap_coefficient', settings.OVERLAP_COEFFICIENT)
        self.overlap = int(self.overlap_coefficient * self.cut_size)

        clean_create_folder(self.anno_location)

    def __call__(self):
        """ Functor call """
        return self.__process_files()

    def __create_image_json_file(self, filename, source_filename, x, y, xmax, ymax):
        """
        Creates a roi json file at self.image_location using the provided
        arguments
        """
        with open(os.path.join(self.image_location, filename), 'w') as _file:
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
        for image_path in tqdm(self.image_list):
            image = plt.imread(image_path)
            h, w = image.shape[:2]
            y = 0

            while y <= (h-self.cut_size):
                x = 0
                while x <= (w-self.cut_size):
                    self.__create_image_json_file(
                        self.__format_clean_filename(image_path, x, y), image_path,
                        x, y,
                        x+self.cut_size, y+self.cut_size
                    )

                    x += self.overlap

                if (x-self.cut_size) <= (self.holdback*self.cut_size):
                    x = w - self.cut_size

                    self.__create_image_json_file(
                        self.__format_clean_filename(image_path, x, y), image_path,
                        x, y,
                        w, y+self.cut_size
                    )

                y += self.overlap

            if ((h/self.cut_size) - (h//self.cut_size)) >= self.holdback:
                x = 0
                y = h - self.cut_size
                while x <= (w-self.cut_size):
                    self.__create_image_json_file(
                        self.__format_clean_filename(image_path, x, y), image_path,
                        x, y,
                        x+self.cut_size, y+self.cut_size
                    )

                    x += self.overlap

                self.__create_image_json_file(
                    self.__format_clean_filename(image_path, w-self.cut_size, h-self.cut_size), image_path,
                    w-self.cut_size, h-self.cut_size,
                    w, h
                )

    def __create_labels(self):
        """
        Reads the file names from self.anno_location direcotry, creates the
        a dictionary with the format: key: filename, value: label; and saves it
        into a labels.pickle at self.anno_location directory
        """
        labels = dict()

        for filename in tqdm(os.listdir(self.image_location)):
            if filename.startswith('b'):
                labels[filename] = Label.BENIGN.id
            elif filename.startswith('is'):
                labels[filename] = Label.INSITU.id
            elif filename.startswith('iv'):
                labels[filename] = Label.INVASIVE.id
            elif filename.startswith('n'):
                labels[filename] = Label.NORMAL.id
            else:
                raise ImageNameInvalid()

        with open(os.path.join(self.anno_location, settings.LABELS_FILENAME), 'w') as file_:
            json.dump(labels, file_)

    def __process_files(self):
        """ Creates the minipatch json files and the labebls.pickle file  """
        print("Creating minipatches...")
        self.__create_minipatches()
        print("Creating labels...")
        self.__create_labels()


class TrainTestSplit:
    """
    Splits the dataset into train and test subsets

    Args:
        root_folder (str): path to the folder containing the json images and labels file
        labels      (str): name of the labels file
        test_size   (float): test dataset size in range [0, 1]

    Usage:
        TrainTestSplit(test_size=0.2)()
    """

    def __init__(self, *args, **kwargs):
        """ Initializes the object and loads the whole labels json file """
        self.root_folder = kwargs.get('root_folder', settings.OUTPUT_FOLDER)
        assert os.path.isdir(self.root_folder)
        self.labels = os.path.join(
            self.root_folder, kwargs.get('labels', settings.LABELS_FILENAME))
        assert os.path.isfile(self.labels)
        self.test_size = kwargs.get('test_size', settings.TEST_SIZE)
        assert isinstance(self.test_size, float)

        with open(self.labels, 'r') as file_:
            self.labels = json.load(file_)

        self.x_train = self.x_test = self.y_train = self.y_test = None

    def __call__(self):
        """
        * Functor call
        * Splits the dataset into train and test subsets
        """
        self.__split_train_test_labels()
        self.__move_train_test_images()

    def __split_train_test_labels(self):
        """
        Splits the dataset and saves the train and test label json files at train and test
        folders respectively
        """
        print("Creating labels json files...")
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            list(self.labels.keys()), list(self.labels.values()), test_size=self.test_size,
            random_state=settings.RANDOM_STATE, stratify=list(self.labels.values())
        )

        folders = [os.path.join(self.root_folder, folder)
                   for folder in [settings.TRAIN_FOLDER_NAME, settings.TEST_FOLDER_NAME]]

        for folder in tqdm(folders):
            clean_create_folder(folder)

            # saving labels file
            if folder == settings.TRAIN_FOLDER_NAME:
                keys, values = self.x_train, self.y_train
            else:
                keys, values = self.x_test, self.y_test

            with open(os.path.join(folder, settings.LABELS_FILENAME), 'w') as file_:
                json.dump(dict(zip(keys, values)), file_)

    def __move_train_test_images(self):
        """ Moves image json files to their corresponding train/test folders """
        print("Creating train dataset")
        for filename in tqdm(self.x_train):
            shutil.move(
                os.path.join(self.root_folder, filename),
                os.path.join(self.root_folder, settings.TRAIN_FOLDER_NAME, filename)
            )

        print("Creating test dataset")
        for filename in tqdm(self.x_test):
            shutil.move(
                os.path.join(self.root_folder, filename),
                os.path.join(self.root_folder, settings.TEST_FOLDER_NAME, filename)
            )


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
        # TODO: review is casting to float is strictly necessary
        target = np.array(self.data.iloc[idx, 1]).astype('float')
        sample = {'image': image, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample


# def read_json_img():
#     """  """
#     img = Image.open()
#     image = cv2.imread(fimg.filename)
#     img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))
#     img, info_img = preprocess(img, self.imgsize, jitter=0)  # info = (h, w, nh, nw, dx, dy)
#     img = np.transpose(img / 255., (2, 0, 1))
#     img = torch.from_numpy(img).float().unsqueeze(0)

#     if use_cuda():
#         img = Variable(img.type(torch.cuda.FloatTensor))
#     else:
#         img = Variable(img.type(torch.FloatTensor))


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
