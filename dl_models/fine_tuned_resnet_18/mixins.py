# -*- coding: utf-8 -*-
""" dl_models/fine_tuned_resnet_18/mixins """

from torchvision import transforms

from dl_models.fine_tuned_resnet_18 import constants as local_constants


class TransformsMixins:
    """ Holds default basic transforms """

    @staticmethod
    def get_default_data_transforms():
        """
        Returns the default data transformations to be appliend to the train and test datasets
        """
        # TODO: Try with the commented transforms
        return {
            'train': transforms.Compose([
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(local_constants.MEAN, local_constants.STD)
            ]),
            'test': transforms.Compose([
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(local_constants.MEAN, local_constants.STD)
            ]),
        }
