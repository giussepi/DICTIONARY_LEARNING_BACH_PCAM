# -*- coding: utf-8 -*-
"""
utils/datasets/bach

Classes and methods to read and process data from ICIAR 2018 BACH challenge

https://iciar2018-challenge.grand-challenge.org/Dataset/
"""

import json
import os
import shutil

import matplotlib.pyplot as plt
from tqdm import tqdm

from constants.constants import Label
from core.exceptions.dataset import ImageNameInvalid
import settings
from utils.files import get_name_and_extension


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

        self.__clean_create_folders()

    def __call__(self):
        """ Functor call """
        return self.__process_files()

    def __clean_create_folders(self):
        """ Removes the output folder and recreate it for the new outputs """
        if os.path.isdir(self.anno_location):
            shutil.rmtree(self.anno_location)
        os.mkdir(self.anno_location)

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

        with open(os.path.join(self.anno_location, 'labels.pickle'), 'w') as file_:
            json.dump(labels, file_)

    def __process_files(self):
        """ Creates the minipatch json files and the labebls.pickle file  """
        print("Creating minipatches...")
        self.__create_minipatches()
        print("Creating labels...")
        self.__create_labels()


def plot_json_img(file_path, figsize=None, save_to_disk=False, folder_path=''):
    """
    Reads a json image file and based on the provided parameters it can be plotted or
    saved to disk.

    Args:
        file_path         (str): relative path to the iamge json file
        figsize (None or tuple): dimensions of the image to be plotted
        save_to_disk     (bool): if true the image is saved to disk, otherwise it's plotted
        folder_path       (str): relative path to the folder where the image will be saved

    Usage:
        plot_json_img(os.path.join(settings.OUTPUT_FOLDER, 'b001_0_0.json'), (9, 9), False)
    """
    assert isinstance(file_path, str)
    assert os.path.isfile(file_path)
    assert isinstance(save_to_disk, bool)
    assert isinstance(folder_path, str)

    if folder_path and not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    if figsize:
        assert isinstance(figsize, tuple)
        assert len(figsize) == 2
        assert figsize[0] > 0 and figsize[1] > 0
    else:
        figsize = (8, 8)

    with open(file_path, 'r') as file_:
        data = json.load(file_)
        plt.figure(figsize=figsize)
        image = plt.imread(data['source'])[
            data['roi']['y']:data['roi']['y']+data['roi']['h'],
            data['roi']['x']:data['roi']['x']+data['roi']['w'],
        ]
        plt.imshow(image)

        if save_to_disk:
            name, _ = get_name_and_extension(file_path.split('/')[-1])
            plt.savefig(os.path.join(folder_path, '{}.png'.format(name)))
        else:
            plt.show()


def plot_n_first_json_images(
        n_images, figsize=None, save_to_disk=False, folder_path='', clean_folder=False):
    """
    Reads the n-fist json images from settings.OUTPUT_FOLDER and based on the provided parameters
    they can be plotted or saved to disk.

    Args:
        n_images          (int): number of images to read
        figsize (None or tuple): dimensions of the image to be plotted
        save_to_disk     (bool): if true the image is saved to disk, otherwise it's plotted
        folder_path       (str): relative path to the folder where the image will be saved
        clean_folder     (bool): if true the folder is deleted and re-created

    Usage:
        plot_n_first_json_images(20, (9, 9), True, 'my_folder', True)
    """
    assert isinstance(n_images, int)
    assert isinstance(clean_folder, bool)

    if clean_folder and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)

    print("Plotting images")
    for image in tqdm(os.listdir(settings.OUTPUT_FOLDER)[:n_images]):
        plot_json_img(
            os.path.join(settings.OUTPUT_FOLDER, image), figsize, save_to_disk, folder_path)
