# -*- coding: utf-8 -*-
"""
utils/datasets/bach

Classes and methods to read and process data from ICIAR 2018 BACH challenge

https://iciar2018-challenge.grand-challenge.org/Dataset/
"""

import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from gtorch_utils.constants import DB
from gutils.datasets.utils import TrainValTestSplit as gutils_TrainValTestSplit
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import rescale, resize
from torch.utils.data import Dataset
from tqdm import tqdm

import settings
from constants.constants import ProcessImageOption
from dl_models.fine_tuned_resnet_18.mixins import TransformsMixins
from utils.datasets.base import BaseTorchDataset, BaseTorchNetDataset
from utils.datasets.mixins import CreateJSONFilesMixin, ReadSplitFileMixin, CreateLabelsMixin
from utils.files import get_name_and_extension
from utils.feature_descriptors.random_faces import RandomFaces as RandFaces
from utils.utils import clean_create_folder, clean_json_filename, get_filename_and_extension,\
    remove_folder


class RescaleResize:
    """
    Creates a rescaled version of BACH dataset

    Usage:
        RescaleResize(.25)()  # rescales using a .25 scaling factor
        RescaleResize((100, 100, 3))()  # resizes to (100, 100, 3)
    """

    def __init__(self, scale, anti_aliasing=True, image_format='tiff', multichannel=True):
        """
        Initializes the object

        Args:
            # skimage.transform.rescale and resize.
            scale (float or tuple of floats or ints): A tuple a integers will perform a resize transformation; otherwise, a rescale operations is performed. See scale and output_shape at https://scikit-image.org/docs/dev/api/skimage.transform.html
            anti_aliasing                     (bool): Whether to apply a Gaussian filter to smooth the image prior to down-scaling.It is crucial to filter when down-sampling the image to avoid aliasing artifacts.
            image_format                       (str): image format
            multichannel                      (bool): Whether the last axis of the image is to be interpreted as multiple channels or another spatial dimension. Only applied when when rescaling.
        """
        assert isinstance(scale, (float, tuple))
        assert isinstance(anti_aliasing, bool)
        assert isinstance(image_format, str)
        assert isinstance(multichannel, bool)

        self.scale = scale
        self.anti_aliasing = anti_aliasing
        self.image_format = image_format
        self.image_extension = image_format if image_format != 'tiff' else 'tif'
        self.image_extension = '.{}'.format(self.image_extension)
        self.transform_kwargs = {'anti_aliasing': self.anti_aliasing}

        if isinstance(scale, tuple) and isinstance(scale[0], int):
            # Resizing
            self.transform = resize
        else:
            # Rescaling
            self.transform = rescale
            self.transform_kwargs['multichannel'] = multichannel

    def __call__(self):
        """ Functor call """
        self.__process()

    def __process(self):
        """
        Creates transformed images and saves them in a directory at the same level of the
        dataset directory
        """
        scaled_path = os.path.join(
            Path(settings.TRAIN_PHOTOS_DATASET).parent,
            '{}_{}'.format(os.path.basename(settings.TRAIN_PHOTOS_DATASET), self.scale)
        )
        remove_folder(scaled_path)

        # creating new images
        for folder in os.listdir(settings.TRAIN_PHOTOS_DATASET):
            current_folder = os.path.join(settings.TRAIN_PHOTOS_DATASET, folder)

            if Path(current_folder).is_dir():
                new_folder = os.path.join(scaled_path, folder)
                clean_create_folder(new_folder)
                print('Creating new images from directory: {}'.format(folder))

                for image_name in tqdm(list(filter(lambda x: x.endswith(self.image_extension), os.listdir(current_folder)))):
                    image = plt.imread(os.path.join(current_folder, image_name))
                    rescaled_img = self.transform(image, self.scale, **self.transform_kwargs)

                    pil_img = Image.fromarray((rescaled_img * 255 / np.max(rescaled_img)).astype(np.uint8))
                    pil_img.save(os.path.join(new_folder, image_name))

        # copyting CSV file
        shutil.copyfile(
            settings.TRAIN_PHOTOS_GROUND_TRUTH,
            os.path.join(scaled_path, os.path.basename(settings.TRAIN_PHOTOS_GROUND_TRUTH))
        )


class BasePrepareDataset(CreateLabelsMixin, ReadSplitFileMixin):
    """
    Provides methods to create ROI JSON files for TIFF images from settings.TRAIN_SPLIT_FILENAME,
    settings.VALID_SPLIT_FILENAME, and settings.TEST_SPLIT_FILENAME files, and saves them at
    settings.TRAIN_FOLDER_NAME, settings.VALID_FOLDER_NAME and settings.TEST_FOLDER_NAME
    directories respectively. Also for creates the labels file for each train and test directory.

    Use it as a base class to create your own ROI JSON files. Just make sure you override the
    _create_json_files method.

    Usage:
        class MyPatches(BasePrepareDataset):
            def _create_json_files(self):
                # some stuff

        MyPatches()()
    """

    def __init__(self, *args, **kwargs):
        """
        * Initializes the object
        * Cleans the train, validation and test output folders (performs delete and create operations)
        """
        self.image_list = None
        self.split_files = [
            settings.TRAIN_SPLIT_FILENAME, settings.VALID_SPLIT_FILENAME, settings.TEST_SPLIT_FILENAME]
        self.folder_names = [
            settings.TRAIN_FOLDER_NAME, settings.VALID_FOLDER_NAME, settings.TEST_FOLDER_NAME]

        for folder in self.folder_names:
            clean_create_folder(os.path.join(settings.OUTPUT_FOLDER, folder))

    def __call__(self):
        """ Functor call """
        self._load_image_list()
        self._create_json_files()
        self._create_labels(self.folder_names)

    def _load_image_list(self):
        """
        Loads the train, validation, test json files, creates the paths to the TIFF images
        into a list, and assings the list to self.image_list.
        """
        self.image_list = []

        for split_filename, split_folder_name in zip(self.split_files, self.folder_names):
            for filename, label in self.read_split_file(split_filename, self.split_files):
                self.image_list.append((
                    os.path.join(settings.TRAIN_PHOTOS_DATASET, label, filename),
                    split_folder_name
                ))

    def _create_image_json_file(self, filename, folder, source_filename, x, y, xmax, ymax):
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
        assert folder in self.folder_names
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
    def _format_clean_filename(filename, x_suffix, y_suffix):
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

    def _create_json_files(self):
        """
        * Processes the WSIs and saves rois as CSV files
        * Must be overridden
        """
        # Example:
        # print("Processing images to create json files")
        # for image_path, folder in tqdm(self.image_list):
        #     image = plt.imread(image_path)
        #     h, w = image.shape[:2]

        #     # You can create the ROI files you want just make sure to used the following
        #     # methods when saving them
        #     self._create_image_json_file(
        #         self._format_clean_filename(image_path, x, y),
        #         folder, image_path, 0, 0, w, h
        #     )
        raise NotImplementedError


class BachTorchDataset(BaseTorchDataset):
    """ BACH torch dataset handler """

    def __init__(self, subset, **kwargs):
        super().__init__(subset, sub_datasets=DB, **kwargs)


class BachTorchNetDataset(BaseTorchNetDataset):
    """ BACH torch dataset handler for models requiring images as inputs """

    def __init__(self, subset, **kwargs):
        super().__init__(subset, sub_datasets=DB, **kwargs)


class MiniPatch(BasePrepareDataset):
    """
    Creates minipatches for each TIFF file at settings.TRAIN_SPLIT_FILENAME,
    settings.VALID_SPLIT_FILENAME and settings.TEST_SPLIT_FILENAME and saves them at
    settings.TRAIN_FOLDER_NAME, settings.VALID_FOLDER_NAME and
    settings.TEST_FOLDER_NAME folders respectively. Also for creates the labels file
    for each train and test folder.

    Usage:
        MiniPatch()()
    """

    def _create_json_files(self):
        """
        Reads the images from self.image_list and creates the
        minipatch json files
        """
        print("Processing images to create minipathes")
        for image_path, folder in tqdm(self.image_list):
            image = plt.imread(image_path)
            h, w = image.shape[:2]
            y = 0

            while y <= (h-settings.CUT_SIZE):
                x = 0
                while x <= (w-settings.CUT_SIZE):
                    self._create_image_json_file(
                        self._format_clean_filename(image_path, x, y), folder, image_path,
                        x, y,
                        x+settings.CUT_SIZE, y+settings.CUT_SIZE
                    )

                    x += settings.OVERLAP

                if (x-settings.CUT_SIZE) <= (settings.HOLDBACK*settings.CUT_SIZE):
                    x = w - settings.CUT_SIZE

                    self._create_image_json_file(
                        self._format_clean_filename(image_path, x, y), folder, image_path,
                        x, y,
                        w, y+settings.CUT_SIZE
                    )

                y += settings.OVERLAP

            if ((h/settings.CUT_SIZE) - (h//settings.CUT_SIZE)) >= settings.HOLDBACK:
                x = 0
                y = h - settings.CUT_SIZE
                while x <= (w-settings.CUT_SIZE):
                    self._create_image_json_file(
                        self._format_clean_filename(image_path, x, y), folder, image_path,
                        x, y,
                        x+settings.CUT_SIZE, y+settings.CUT_SIZE
                    )

                    x += settings.OVERLAP

                self._create_image_json_file(
                    self._format_clean_filename(image_path, w-settings.CUT_SIZE, h-settings.CUT_SIZE),
                    folder, image_path,
                    w-settings.CUT_SIZE, h-settings.CUT_SIZE,
                    w, h
                )


class WholeImage(CreateJSONFilesMixin, BasePrepareDataset):
    """
    Creates JSON files covering the whole TIFF images from settings.TRAIN_SPLIT_FILENAME,
    settings.VALID_SPLIT_FILENAME and settings.TEST_SPLIT_FILENAME files, and saves them
    at settings.TRAIN_FOLDER_NAME and settings.TEST_FOLDER_NAME folders respectively.
    Also for creates the labels file for each train and test folder.

    Use it to work with the whole images instead of patches.

    Usage:
        WholeImage()()
    """


class SelectNRandomPatches(CreateLabelsMixin, ReadSplitFileMixin):
    """
    Performs the selection N minipatches for each subdataset.

    WARNING: it deletes all the non-selected patches and overrides their label files

    Usage:
        SelectNRandomPatches(100)()
    """

    def __init__(self, num_patches):
        """
        Initializes the object

        Args:
            num_patches (int): number of patches to be selected per image
        """
        assert isinstance(num_patches, int)
        assert num_patches > 0
        self.num_patches = num_patches
        self.image_list = None
        self.split_files = [
            settings.TRAIN_SPLIT_FILENAME, settings.VALID_SPLIT_FILENAME, settings.TEST_SPLIT_FILENAME]
        self.folder_names = [
            settings.TRAIN_FOLDER_NAME, settings.VALID_FOLDER_NAME, settings.TEST_FOLDER_NAME]

    def __call__(self):
        """ functor call """
        self.select_patches()
        self._create_labels(self.folder_names)

    def select_patches(self):
        """ Selects the patches and removes the original subdataset labels files """

        print("Selecting minipatches from subdatastes")
        for split_filename, split_folder_name in tqdm(tuple(zip(self.split_files, self.folder_names))):
            patches = os.listdir(os.path.join(settings.OUTPUT_FOLDER, split_folder_name))

            for filename, _ in self.read_split_file(split_filename, self.split_files):
                name, _ = get_filename_and_extension(filename)
                file_patches = set(filter(lambda x: x.startswith('{}_'.format(name)), patches))
                chosen_patches = set(random.sample(file_patches, k=self.num_patches))
                patches_to_delete = file_patches.difference(chosen_patches)

                for patch in patches_to_delete:
                    os.remove(os.path.join(settings.OUTPUT_FOLDER, split_folder_name, patch))

            # removing old labels file
            os.remove(os.path.join(
                settings.OUTPUT_FOLDER, split_folder_name, settings.LABELS_FILENAME))


class TrainValTestSplit:
    """
    Splits the dataset into train, validation and test and saves them in CSV files

    Args:
        test_size   (float): test dataset size in range [0, 1]
        val_size    (float): validation dataset size in range [0, 1]

    Usage:
        TrainValTestSplit()()
    """

    def __init__(self, *args, **kwargs):
        """ Initializes the instance """
        self.test_size = kwargs.get('test_size', settings.TEST_SIZE)
        self.val_size = kwargs.get('val_size', settings.VAL_SIZE)
        assert isinstance(self.test_size, (float, int))
        assert isinstance(self.val_size, (float, int))
        self.train_xy = self.val_xy = self.test_xy = None

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

        print("Splitting dataset with test = {} & val = {}".format(self.test_size, self.val_size))

        with tqdm(total=1) as pbar:
            x_train, x_val, x_test, y_train, y_val, y_test = gutils_TrainValTestSplit(
                ground_truth[:, 0], ground_truth[:, 1], val_size=self.val_size,
                test_size=self.test_size, random_state=settings.RANDOM_STATE,
                shuffle=True, stratify=ground_truth[:, 1]
            )()

            self.train_xy = np.hstack((
                np.expand_dims(x_train, axis=1), np.expand_dims(y_train, axis=1)))
            self.val_xy = np.hstack((
                np.expand_dims(x_val, axis=1), np.expand_dims(y_val, axis=1)))
            self.test_xy = np.hstack((
                np.expand_dims(x_test, axis=1), np.expand_dims(y_test, axis=1)))
            pbar.update(1)

    def __create_json_files(self):
        """ Saves train, validation and test datasets into JSON files """
        print("Saving train/validation/test dataset into JSON files...")

        file_paths = [
            os.path.join(settings.OUTPUT_FOLDER, filename)
            for filename in [
                settings.TRAIN_SPLIT_FILENAME,
                settings.VALID_SPLIT_FILENAME,
                settings.TEST_SPLIT_FILENAME
            ]
        ]

        clean_create_folder(settings.OUTPUT_FOLDER)

        for file_path, data in tqdm(zip(file_paths, (self.train_xy, self.val_xy, self.test_xy))):
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

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'target': target}


class BaseDatasetCreator(TransformsMixins):
    """ Holds basic handles to create a dataset """

    def __init__(self, *args, **kwargs):
        """
        Initializes the instance
        Kwargs:
            data_transforms      (dict): data transformations to be applied. See TransformsMixins definition
            codes_folder          (str): folder to store the generated codes
            process_method  (LabelItem): [Optional] processing option. See constants.constants.ProcessImageOption
            label_class          (type): Label class (see constants/constants.py)
            sub_datasets (object class): Class holding subdataset information.
                                         See constants.constants.PCamSubDataset and gtorch_utils.constants.DB
        """
        # TODO: verify the torch data adjustment is not necessary here
        self.data_transforms = kwargs.get('data_transforms', self.get_default_data_transforms())
        self.label_class = kwargs.get('label_class', '')
        assert isinstance(self.label_class, type)
        self.codes_folder = kwargs.get('codes_folder', '')
        self.process_method = kwargs.get('process_method', ProcessImageOption.MEAN)
        sub_datasets = kwargs.get('sub_datasets', DB)
        assert isinstance(self.data_transforms, dict)
        assert isinstance(self.codes_folder, str)
        assert self.codes_folder != ''
        assert ProcessImageOption.is_valid_option(self.process_method)
        assert hasattr(sub_datasets, 'SUB_DATASETS')
        self.sub_datasets = sub_datasets.SUB_DATASETS

        # TODO: maybe the transform should happen at the very end, not before calculating
        #       sift descriptors
        self.image_datasets = {
            x: BACHDataset(
                os.path.join(settings.OUTPUT_FOLDER, x), transform=self.data_transforms[x])
            for x in self.sub_datasets
        }

        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.image_datasets[x], batch_size=settings.BATCH_SIZE,
                shuffle=True, num_workers=settings.NUM_WORKERS)
            for x in self.sub_datasets
        }
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in self.sub_datasets}

    def process_input(self, input_):
        """
        Processes the input_ image based on self.process_method and returns the result

        Args:
            input_ (np.ndarray): numpy array
        """
        assert isinstance(input_, np.ndarray)

        if self.process_method.id == ProcessImageOption.CONCATENATE.id:
            return np.c_[input_[0], input_[1], input_[2]].ravel()

        if self.process_method.id == ProcessImageOption.GRAYSCALE.id:
            return rgb2gray(np.moveaxis(input_, [0, 1, 2], [2, 0, 1])).ravel()

        return np.mean(input_, axis=0).ravel()

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
        # assert dataset in self.sub_datasets
        # assert isinstance(formatted_data, dict)

        #
        # for data in tqdm(self.dataloaders[dataset]):
        #     inputs = data['image'].numpy()
        #     labels = data['target'].numpy()

        #     for input_, label in zip(inputs, labels):
        #         # Do something cool
        #         processed_input = self.process_input(input_) # or define your own way
        #         formatted_data['codes'].append(processed_input.tolist())
        #         formatted_data['labels'].append(label.tolist())

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
            (len(self.label_class.CHOICES), formatted_data['labels'].shape[0]), dtype=float)

        for index, label_item in enumerate(self.label_class.CHOICES):
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
        for dataset in self.sub_datasets:
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
        from gtorch_utils.constants import DB
        from constants.constants import ProcessImageOption, Label, PCamLabel, PCamSubDataset

        # for BACH
        ri = RawImages(
            process_method=ProcessImageOption.GRAYSCALE, label_class=Label, sub_datasets=DB)
        # for PatchCamelyon
        ri = RawImages(
            process_method=ProcessImageOption.GRAYSCALE, label_class=PCamLabel, sub_datasets=PCamSubDataset)

        ri.create_datasets_for_LC_KSVD('my_raw_dataset.json')
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the instance

        Kwargs:
            process_method (LabelItem): [Optional] processing option. See constants.constants.ProcessImageOption
        """
        super().__init__(*args, codes_folder=settings.RAW_CODES_FOLDER, **kwargs)

    def process_data(self, dataset, formatted_data):
        """
        Processes the data properly and places it in formatted_data.

        Args:
            dataset         (str): sub-dataset name
            formatted_data (dict): dictionary to store all codes and labels
        """
        assert dataset in self.sub_datasets
        assert isinstance(formatted_data, dict)

        for data in tqdm(self.dataloaders[dataset]):
            inputs = data['image'].numpy()
            labels = data['target'].numpy()

            for input_, label in zip(inputs, labels):
                processed_input = self.process_input(input_)
                formatted_data['codes'].append(processed_input.tolist())
                formatted_data['labels'].append(label.tolist())

        formatted_data['codes'] = np.array(formatted_data['codes'])
        formatted_data['labels'] = np.array(formatted_data['labels'])


class RandomFaces(BaseDatasetCreator):
    """
    Creates a dataset for LC-KSVD using random face descriptors

    Usage:
        from gtorch_utils.constants import DB
        from constants.constants import ProcessImageOption, Label, PCamLabel, PCamSubDataset

        # for BACH
        randfaces = RandomFaces(img_height=512, img_width=512, process_method=ProcessImageOption.GRAYSCALE, label_class=Label, sub_datasets=DB)
        # for PatchCamelyon
        randfaces = RandomFaces(img_height=32, img_width=32, process_method=ProcessImageOption.GRAYSCALE, label_class=PCamLabel, sub_datasets=PCamSubDataset)

        randfaces.create_datasets_for_LC_KSVD('randfaces_dataset.json')
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the instance
        Kwargs:
            process_method (LabelItem): [Optional] processing option. See constants.constants.ProcessImageOption
            img_height       (int): height of the images
            img_width        (int): width of images
            fd_dimension     (int): dimension of random-face feature descriptor
        """
        super().__init__(*args, codes_folder=settings.RANDOM_FACE_FOLDER, **kwargs)
        self.img_height = kwargs.get('img_height', '')
        self.img_width = kwargs.get('img_width', '')
        self.fd_dimension = kwargs.get('fd_dimension', settings.FD_DIMENSION)
        self.concat_channels = self.process_method.id == ProcessImageOption.CONCATENATE.id
        self.randfaces_descriptor = RandFaces(self.img_height, self.img_width,
                                              self.fd_dimension, self.concat_channels)

    def process_data(self, dataset, formatted_data):
        """
        Processes the data properly and places it in formatted_data.

        Args:
            dataset         (str): sub-dataset name
            formatted_data (dict): dictionary to store all codes and labels
        """
        assert dataset in self.sub_datasets
        assert isinstance(formatted_data, dict)

        for data in tqdm(self.dataloaders[dataset]):
            inputs = data['image'].numpy()
            labels = data['target'].numpy()

            for input_, label in zip(inputs, labels):
                processed_input = self.process_input(input_)
                formatted_data['codes'].append(
                    self.randfaces_descriptor.get_feature_descriptor(processed_input).tolist())
                formatted_data['labels'].append(label.tolist())

        formatted_data['codes'] = np.array(formatted_data['codes'])
        formatted_data['labels'] = np.array(formatted_data['labels'])


class SparseCodes(BaseDatasetCreator):
    """
    Creates a dataset for LC-KSVD using sparse codes

    Usage:
        from gtorch_utils.constants import DB
        from lcksvd.dksvd import DKSVD
        from constants.constants import ProcessImageOption, Label, PCamLabel, PCamSubDataset

        # for BACH
        ri = SparseCodes(
            process_method=ProcessImageOption.GRAYSCALE, label_class=Label, sub_datasets=DB,
            sparse_coding=DKSVD.get_sparse_representations,
            sparse_coding_kwargs=dict(D=np.load('D.npy'), sparsitythres=15)
        )
        # for PatchCamelyon
        ri = SparseCodes(
            process_method=ProcessImageOption.GRAYSCALE, label_class=PCamLabel, sub_datasets=PCamSubDataset
            sparse_coding=DKSVD.get_sparse_representations,
            sparse_coding_kwargs=dict(D=np.load('D.npy'), sparsitythres=15)
        )

        ri.create_datasets_for_LC_KSVD('sparse_codes_dataset.json')
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the instance

        Kwargs:
            process_method (LabelItem): Optional processing option. See constants.constants.ProcessImageOption
            sparse_coding (function/method): Sparse coding implementation
            sparse_coding_kwargs (dict): Arguments for the sparse_coding implementation
        """
        super().__init__(*args, codes_folder=settings.SPARSE_CODES_FOLDER, **kwargs)
        self.sparse_coding = kwargs.get('sparse_coding')
        self.sparse_coding_kwargs = kwargs.get('sparse_coding_kwargs')

    def process_data(self, dataset, formatted_data):
        """
        Processes the data properly and places it in formatted_data.

        Args:
            dataset         (str): sub-dataset name
            formatted_data (dict): dictionary to store all codes and labels
        """
        assert dataset in self.sub_datasets
        assert isinstance(formatted_data, dict)

        for data in tqdm(self.dataloaders[dataset]):
            inputs = data['image'].numpy()
            labels = data['target'].numpy()

            for input_, label in zip(inputs, labels):
                processed_input = self.process_input(input_)
                learned_representation = self.sparse_coding(
                    processed_input, **self.sparse_coding_kwargs)
                formatted_data['codes'].append(learned_representation.tolist())
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
        if not image.flags['WRITEABLE']:
            image = image.copy()

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
        if image != settings.LABELS_FILENAME:
            plot_json_img(
                os.path.join(read_folder_path, image), figsize, save_to_disk, save_folder_path,
                carousel, remove_axes, dpi
            )
