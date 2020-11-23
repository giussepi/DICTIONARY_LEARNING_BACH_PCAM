# -*- coding: utf-8 -*-
""" dl_models/fine_tuned_resnet_18/mixins """

from torchvision import transforms

from gtorch_utils.constants import DB

from dl_models.fine_tuned_resnet_18 import constants as local_constants
from constants.constants import PCamSubDataset, SubDataset


class TransformsMixins:
    """ Holds default basic transforms """

    @staticmethod
    def get_default_data_transforms(torch_pretrained=False):
        """
        Returns the default data transformations to be appliend to the train and test datasets

        Args:
            torch_pretrained (bool): whether or not apply the adjust suggested when using
            torchvision pre-trained models

        Returns:
            dict(subdataset1=transforms1, subdataset2=transforms2, ...)
        """
        # TODO: Try with the commented transforms
        train_transforms = [
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        test_val_transforms = [
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]

        if torch_pretrained:
            train_transforms.append(transforms.Normalize(local_constants.MEAN, local_constants.STD))
            test_val_transforms.append(transforms.Normalize(local_constants.MEAN, local_constants.STD))

        return {
            PCamSubDataset.TRAIN: transforms.Compose(train_transforms),
            PCamSubDataset.VALIDATION: transforms.Compose(test_val_transforms),
            PCamSubDataset.TEST: transforms.Compose(test_val_transforms),
            DB.VALIDATION: transforms.Compose(test_val_transforms),
        }
