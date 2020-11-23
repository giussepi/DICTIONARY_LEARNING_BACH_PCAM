# -*- coding: utf-8 -*-
""" dl_models/fine_tuned_resnet_18/model """

from __future__ import print_function, division

import copy
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from gutils.decorators import timing
from gtorch_utils.constants import DB
from torch.optim import lr_scheduler
from torchvision import models
from tqdm import tqdm

from constants.constants import Label
from dl_models.fine_tuned_resnet_18 import constants as local_constants
from dl_models.fine_tuned_resnet_18.mixins import TransformsMixins
import settings
from utils.datasets.bach import BACHDataset
from utils.utils import get_filename_and_extension, clean_json_filename


class TransferLearningResnet18(DB, TransformsMixins):
    """"
    Manages the resnet18 by applying transfer learning and optionally fine tuning

    Inspired on: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

    Usage:
        model = TransferLearningResnet18(fine_tune=True)
        model.training_data_plot_grid()
        model.train(num_epochs=25)
        model.save('mymodel.pt')
        model.visualize_model()
        model.test()

        model2 = TransferLearningResnet18(fine_tune=True)
        model2.load('resnet18_fine_tuned.pt')
        model.visualize_model()
        model2.test()
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the instance attributes

        Args:
            data_transforms (dict): for its structure see get_default_data_transforms method
            device  (torch.device): device were model will executed
            fine_tune       (bool): whether perform fine-tuning or use the ConvNet as a fixed feature extractor
        """
        self.data_transforms = kwargs.get(
            'data_transforms', self.get_default_data_transforms(torch_pretrained=True))
        assert isinstance(self.data_transforms, dict)
        self.device = kwargs.get('device', torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        assert isinstance(self.device, torch.device)
        self.fine_tune = kwargs.get('fine_tune', False)
        assert isinstance(self.fine_tune, bool)
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

        self.init_model()

    def init_model(self):
        """ Initializes the model for fine tuning or as a fixed feature extractor """
        self.model = models.resnet18(pretrained=True)

        # If want to use the ConvNet as fixed feature extractor (no fine tuning)
        if not self.fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False

        self.num_ftrs = self.model.fc.in_features

        # Changing last layer
        self.model.fc = nn.Linear(self.num_ftrs, len(Label.CHOICES))
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        if self.fine_tune:
            # Observe that all parameters are being optimized
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        else:
            # Observe that only parameters of final layer are being optimized as
            # opposed to before.
            self.optimizer = optim.SGD(self.model.fc.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    @staticmethod
    def imshow(inp, title=None):
        """
        Imshow for Tensor

        Args:
            inp            (torch.Tensor): Tensor image
            title (list or tuple or None): Image title
        """
        assert isinstance(inp, torch.Tensor)
        assert isinstance(title, (list, tuple)) or title is None

        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array(local_constants.MEAN)
        std = np.array(local_constants.STD)
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    def training_data_plot_grid(self):
        """ Gets a batch of training data and plots a grid  """
        inputs, classes = next(iter(self.dataloaders[self.TRAIN])).values()

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)

        self.imshow(out, title=[Label.get_name(x.item()) for x in classes])

    def load(self, state_dict_path):
        """
        Reads and loads the model state dictionary provided. It must be placed
        in the folder defined at settings.MODEL_SAVE_FOLDER
        """
        path = os.path.join(settings.MODEL_SAVE_FOLDER, state_dict_path)
        assert os.path.isfile(path)

        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    def save(self, filename):
        """
        Saves the model in a file <filename>.pt at settings.MODEL_SAVE_FOLDER

        Args:
            filename (str): filename with '.pt' extension
        """
        if not os.path.isdir(settings.MODEL_SAVE_FOLDER):
            os.makedirs(settings.MODEL_SAVE_FOLDER)

        assert isinstance(filename, str)
        assert filename.endswith('.pt')

        torch.save(self.model.state_dict(), os.path.join(settings.MODEL_SAVE_FOLDER, filename))

    @timing
    def train(self, num_epochs=25):
        """
        * Trains and evaluates the model
        * Sets the best model to self.model

        Args:
            num_epochs (int): number of epochs
        """
        assert isinstance(num_epochs, int)
        assert num_epochs > 0

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in [self.TRAIN, self.VALIDATION]:
                if phase == self.TRAIN:
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for data in self.dataloaders[phase]:
                    inputs = data['image'].to(self.device)
                    labels = data['target'].to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == self.TRAIN):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == self.TRAIN:
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == self.TRAIN:
                    self.scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == self.VALIDATION and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        print('Best test Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)

    @timing
    def test(self):
        """
        Test the model and prints the results
        """
        self.model.eval()
        corrects = 0

        for data in self.dataloaders[self.TEST]:
            inputs = data['image'].to(self.device)
            labels = data['target'].to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

            corrects += torch.sum(preds == labels.data)

        accuracy = corrects.double() / self.dataset_sizes[self.TEST]
        print('Acc: {:.4f}'.format(accuracy))

    def get_CNN_codes(self, sub_dataset):
        """
        Get and returns the CNN codes concatenated from the avgpool layer, also returns
        the labels for each CNN code

        Args:
            sub_dataset (str): any value from self.SUB_DATASETS

        Returns:
            cnn_codes (torch.Tensor), labels (torch.Tensor)
        """
        assert sub_dataset in self.SUB_DATASETS

        cnn_codes = []
        all_labels = []

        def hook(module, input_, output_):
            cnn_codes.append(output_)

        self.model.avgpool.register_forward_hook(hook)
        self.model.eval()
        print("Processing mini-patch batches to get CNN codes from {} sub-dataset"
              .format(sub_dataset))

        for data in tqdm(self.dataloaders[sub_dataset]):
            inputs = data['image'].to(self.device)
            all_labels.append(data['target'])

            with torch.no_grad():
                self.model(inputs)

        return torch.cat(cnn_codes, dim=0), torch.cat(all_labels)

    def get_all_CNN_codes(self):
        """
        Creates and returns a dictionary containing  all the CNN codes and labels for all
        the sub-datasets created

        Returns:
            {'sub_dataset_1': [cnn codes torch.Tensor, labels torch.Tensor], ...}
        """
        return dict(
            (sub_dataset, self.get_CNN_codes(sub_dataset))
            for sub_dataset in self.SUB_DATASETS
        )

    def format_for_LC_KSVD(self, sub_dataset, cnn_codes, labels, save=False, filename=''):
        """
        Returns a dictionary with cnn_codes and labels for the sub_dataset chosen. Optionally,
        it saves the dictionary in the file <filename>_<sub_dataset>.json at
        settings.CNN_CODES_FOLDER

        Args:
            sub_dataset        (str): Any value from self.SUB_DATASETS
            cnn_codes (torch.Tensor): Tensor with all cnn codes.
            labels    (torch.Tensor): Tensor with all labels.

            save              (bool): Whether or not save the result
            filename           (str): Filename with .json extension

        Returns:
            {'<sub_dataset>': [cnn codes list of lists, labels list]}

        """
        assert sub_dataset in self.SUB_DATASETS
        assert isinstance(cnn_codes, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert isinstance(save, bool)

        cleaned_filename = clean_json_filename(filename)

        # Workaround to serialize as JSON the numpy arrays
        formatted_cnn_codes = cnn_codes.squeeze().T.cpu().numpy()
        # TODO: review if it's necessary to use float
        formatted_labels = np.zeros((len(Label.CHOICES), labels.shape[0]), dtype=float)

        for index, label_item in enumerate(Label.CHOICES):
            formatted_labels[index, labels == label_item.id] = 1

        # Workaround to serialize numpy arrays as JSON
        formatted_data = {
            'cnn_codes': formatted_cnn_codes.tolist(),
            'labels': formatted_labels.tolist()
        }

        if save:
            if not os.path.isdir(settings.CNN_CODES_FOLDER):
                os.makedirs(settings.CNN_CODES_FOLDER)

            with open(os.path.join(settings.CNN_CODES_FOLDER, cleaned_filename), 'w') as file_:
                json.dump(formatted_data, file_)

        return formatted_data

    def format_all_for_LC_KSVD(self, cnn_codes_labels, save=False, filename=''):
        """
        Returns a dictionary containing all the cnn_codes and labels for each sub-dataset
        created properly formatted to be used by the LC-KSVD algorithm. Optionally, it
        saves the dictionary splitted in several files with the
        format <filename>_<sub_dataset>.json at settings.CNN_CODES_FOLDER

        Args:
            cnn_codes_labels (dict): Dictionary returned by the get_all_CNN_codes method
            save             (bool): Whether or not save the result
            filename          (str): filename with .json extension

        Returns:
            {'sub_dataset_1': [cnn codes list of lists, labels list], ...}
        """
        assert isinstance(cnn_codes_labels, dict)
        assert isinstance(save, bool)

        cleaned_filename = clean_json_filename(filename)
        name, extension = get_filename_and_extension(cleaned_filename)

        formatted_data = dict()

        print("Formatting and saving sub-datasets CNN codes for LC-KSVD")
        for sub_dataset in tqdm(self.SUB_DATASETS):
            new_name = '{}_{}.{}'.format(name, sub_dataset, extension)
            formatted_data[sub_dataset] = self.format_for_LC_KSVD(
                sub_dataset, *cnn_codes_labels[sub_dataset], save, new_name)

        return formatted_data

    def create_datasets_for_LC_KSVD(self, filename):
        """
        * Gets all CNN codes and labels
        * Transform the data to be compatible with the LC-KSVD algorithm
        * Saves dataset at settings.CNN_CODES_FOLDER using several files named
          <filename>_<sub_dataset>.json

        Note: The models must be trained. So call the 'train' method or load
              the weights.

        Args:
            filename (str): filename with .json extension

        Usage:
            model = TransferLearningResnet18(fine_tune=True)
            model.load('fine_tuned_resnet18.pt')
            model.create_datasets_for_LC_KSVD('my_dataset.json')
        """
        all_cnn_codes = self.get_all_CNN_codes()
        self.format_all_for_LC_KSVD(all_cnn_codes, save=True, filename=filename)

    def visualize_model(self, num_images=6):
        """
        Plots some images with its predicitons

        Args:
            num_images (int) : number of images to plot
        """
        assert isinstance(num_images, int)
        assert num_images > 0

        was_training = self.model.training
        self.model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, data in enumerate(self.dataloaders[self.TEST]):
                inputs = data['image'].to(self.device)
                labels = data['target'].to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(Label.get_name(preds[j].item())))
                    self.imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        self.model.train(mode=was_training)
                        return

            self.model.train(mode=was_training)
