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
from dl_models.fine_tuned_resnet_18.mixins import TransformsMixins
import settings
from utils.files import get_name_and_extension
from utils.feature_descriptors.random_faces import RandomFaces as RandFaces
from utils.utils import clean_create_folder, clean_json_filename, get_filename_and_extension


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
        assert isinstance(self.test_size, (float, int))
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


class BaseDatasetCreator(TransformsMixins):
    """ Holds basic handles to create a dataset """
    TRAIN = 'train'
    TEST = 'test'

    SUB_DATASETS = [TRAIN, TEST]

    def __init__(self, *args, **kwargs):
        """
        Initializes the instance
        Kwargs:
            data_transforms (dict): data transformations to be applied. See TransformsMixins definition
            codes_folder     (str): folder to store the generated codes
        """
        self.data_transforms = kwargs.get('data_transforms', self.get_default_data_transforms())
        self.codes_folder = kwargs.get('codes_folder', '')
        assert isinstance(self.data_transforms, dict)
        assert isinstance(self.codes_folder, str)
        assert self.codes_folder != ''
        self.image_datasets = {
            x: BACHDataset(
                os.path.join(settings.OUTPUT_FOLDER, x), transform=self.data_transforms[x])
            for x in self.SUB_DATASETS
        }
        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.image_datasets[x], batch_size=settings.BATCH_SIZE,
                shuffle=True, num_workers=settings.NUM_WORKERS)
            for x in self.SUB_DATASETS
        }
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in self.SUB_DATASETS}

    def process_data(self, dataset, formatted_data):
        """
        * Processes the data properly and places it in formatted_data.
        * Must be overridden

        Args:
            dataset         (str): sub-dataset name
            formatted_data (dict): dictionary to store all codes and labels

        NOTE:
            formatted_data['code'] must be a numpy array with shape [num_images, code_length]
            formatted_data['code'] must be a numpy array with shape [num_images, ]
        """
        # Example:
        # assert dataset in self.SUB_DATASETS
        # assert isinstance(formatted_data, dict)

        # counter = 0
        #
        # for data in tqdm(self.dataloaders[dataset]):
        #     inputs = data['image'].numpy()
        #     labels = data['target'].numpy()

        #     for input_, label in zip(inputs, labels):
        #         # Do something cool
        #         processed_input = ...
        #         formatted_data['codes'].append(processed_input)
        #         formatted_data['labels'].append(label_)
        #         counter += 1

        #     # Don't forget to provide numpy arrays
        #     formatted_data['codes'] = np.array(formatted_data['codes'])
        #     formatted_data['labels'] = np.array(formatted_data['labels'])

        raise NotImplementedError

    def format_for_LC_KSVD(self, formatted_data):
        """
        Formats data from the provided dictionary to be compatible with LC-KSVD algorithm

        Args:
            formatted_data (dict): dictionary with codes and labels
        """
        assert isinstance(formatted_data, dict)
        assert isinstance(formatted_data['codes'], np.ndarray)
        assert isinstance(formatted_data['labels'], np.ndarray)

        formatted_data['codes'] = formatted_data['codes'].T
        formatted_labels = np.zeros(
            (len(Label.CHOICES), formatted_data['labels'].shape[0]), dtype=float)

        for index, label_item in enumerate(Label.CHOICES):
            formatted_labels[index, formatted_data['labels'] == label_item.id] = 1

        # Workaround to serialize numpy arrays as JSON
        formatted_data['codes'] = formatted_data['codes'].tolist()
        formatted_data['labels'] = formatted_labels.tolist()

    def create_datasets_for_LC_KSVD(self, filename):
        """
        Args:
            filename (str): filename with .json extension

        Usage:
            model.create_datasets_for_LC_KSVD('my_dataset.json')
        """
        clean_create_folder(self.codes_folder)
        cleaned_filename = clean_json_filename(filename)
        name, extension = get_filename_and_extension(cleaned_filename)

        print("Formatting and saving sub-datasets codes for LC-KSVD")
        for dataset in self.SUB_DATASETS:
            print("Processing image's batches from sub-dataset: {}".format(dataset))
            new_name = '{}_{}.{}'.format(name, dataset, extension)
            formatted_data = {'codes': [], 'labels': []}
            self.process_data(dataset, formatted_data)
            self.format_for_LC_KSVD(formatted_data)

            with open(os.path.join(self.codes_folder, new_name), 'w') as file_:
                json.dump(formatted_data, file_)


class RawImages(BaseDatasetCreator):
    """
    Creates a dataset for LC-KSVD using raw data

    Usage:
    ri = RawImages()
    ri.create_datasets_for_LC_KSVD('my_raw_dataset.json')
    """

    def __init__(self, *args, **kwargs):
        """ Initializes the instance """
        super().__init__(*args, codes_folder=settings.RAW_CODES_FOLDER, **kwargs)

    def process_data(self, dataset, formatted_data):
        """
        Processes the data properly and places it in formatted_data.

        Args:
            dataset         (str): sub-dataset name
            formatted_data (dict): dictionary to store all codes and labels
        """
        assert dataset in self.SUB_DATASETS
        assert isinstance(formatted_data, dict)

        dataset_folder = os.path.join(self.codes_folder, dataset)
        clean_create_folder(dataset_folder)

        counter = 0

        for data in tqdm(self.dataloaders[dataset]):
            inputs = data['image']
            inputs = inputs.view(*inputs.size()[:2], -1).numpy()
            # inputs.shape [32, 3, 1024]
            labels = data['target'].numpy()

            for input_, label in zip(inputs, labels):
                json_filename = '{}.json'.format(counter)
                formatted_data['codes'].append(json_filename)
                formatted_data['labels'].append(label.tolist())

                with open(os.path.join(dataset_folder, json_filename), 'w') as file_:
                    json.dump(np.mean(input_, axis=0).tolist(), file_)
                    # np.mean(input_, axis=0).shape (1024)

                counter += 1

        formatted_data['codes'] = np.array(formatted_data['codes'])
        formatted_data['labels'] = np.array(formatted_data['labels'])


class RandomFaces(BaseDatasetCreator):
    """
    Creates a dataset for LC-KSVD using random face descriptors

    Usage:
        randfaces = RandomFaces(img_height=512, img_width=512, concat_channels=False)
        randfaces.create_datasets_for_LC_KSVD('my_raw_dataset.json')
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the instance
        Kwargs:
            img_height       (int): height of the images
            img_width        (int): width of images
            fd_dimension     (int): dimension of random-face feature descriptor
            concat_channels (bool): if True the image channels are concatenated,
                                    else their mean is used
        """
        super().__init__(*args, codes_folder=settings.RANDOM_FACE_FOLDER, **kwargs)
        self.img_height = kwargs.get('img_height', '')
        self.img_width = kwargs.get('img_width', '')
        self.fd_dimension = kwargs.get('fd_dimension', settings.FD_DIMENSION)
        self.concat_channels = kwargs.get('concat_channels', False)
        assert isinstance(self.concat_channels, bool)
        self.randfaces_descriptor = RandFaces(self.img_height, self.img_width,
                                              self.fd_dimension, self.concat_channels)

    def process_data(self, dataset, formatted_data):
        """
        Processes the data properly and places it in formatted_data.

        Args:
            dataset         (str): sub-dataset name
            formatted_data (dict): dictionary to store all codes and labels
        """
        assert dataset in self.SUB_DATASETS
        assert isinstance(formatted_data, dict)

        for data in tqdm(self.dataloaders[dataset]):
            inputs = data['image'].numpy()
            labels = data['target'].numpy()

            for input_, label in zip(inputs, labels):
                if self.concat_channels:
                    processed_input = np.c_[input_[0], input_[1], input_[2]].ravel()
                else:
                    processed_input = np.mean(input_, axis=0).ravel()

                formatted_data['codes'].append(
                    self.randfaces_descriptor.get_feature_descriptor(processed_input).tolist())
                formatted_data['labels'].append(label.tolist())

        formatted_data['codes'] = np.array(formatted_data['codes'])
        formatted_data['labels'] = np.array(formatted_data['labels'])


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


def plot_json_img(
        file_path, figsize=None, save_to_disk=False, folder_path='', carousel=False,
        remove_axes=False, dpi=100
):
    """
    Reads a json image file and based on the provided parameters it can be plotted or
    saved to disk.

    Args:
        file_path         (str): relative path to the iamge json file
        figsize (None or tuple): dimensions of the image to be plotted
        save_to_disk     (bool): if true the image is saved to disk, otherwise it's plotted
        folder_path       (str): relative path to the folder where the image will be saved
        carousel         (bool): shows images consecutively only if it has been called through plot_n_first_json_images
        remove_axes      (bool): removes the axes and plots only the image without white borders
        dpi               (int): image resolution

    Usage:
        plot_json_img(os.path.join(settings.OUTPUT_FOLDER, 'b001_0_0.json'), (9, 9), False)
    """
    assert isinstance(file_path, str)
    assert os.path.isfile(file_path)
    assert isinstance(save_to_disk, bool)
    assert isinstance(folder_path, str)
    assert isinstance(carousel, bool)
    assert isinstance(remove_axes, bool)
    assert isinstance(dpi, int)
    assert dpi > 0

    if folder_path and not os.path.isdir(folder_path) and save_to_disk:
        os.mkdir(folder_path)

    if figsize:
        assert isinstance(figsize, tuple)
        assert len(figsize) == 2
        assert figsize[0] > 0 and figsize[1] > 0
    else:
        figsize = (8, 8)

    if remove_axes:
        image = read_roi_image(file_path)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image)
    else:
        plt.figure(figsize=figsize, dpi=dpi)
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
        clean_folder=False, carousel=False, remove_axes=False, dpi=100):
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
        remove_axes      (bool): removes the axes and plots only the image without white borders
        dpi               (int): image resolution

    Usage:
        plot_n_first_json_images(5, os.path.join(settings.OUTPUT_FOLDER, settings.TRAIN_FOLDER_NAME),
                                (9, 9), False, 'my_folder', False, True)
    """
    assert isinstance(n_images, int)
    assert isinstance(clean_folder, bool)

    if clean_folder and os.path.isdir(save_folder_path) and save_to_disk:
        shutil.rmtree(save_folder_path)

    print("Plotting images")
    for image in tqdm(os.listdir(read_folder_path)[:n_images]):
        plot_json_img(
            os.path.join(read_folder_path, image), figsize, save_to_disk, save_folder_path, carousel, remove_axes, dpi)
