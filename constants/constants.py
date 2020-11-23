# -*- coding: utf-8 -*-
""" constants.constants """

from collections import namedtuple

import settings
from core.exceptions.dataset import LabelIdInvalid, PCamLabelIdInvalid

# Colour to plot signals from each label
# See https://matplotlib.org/3.2.1/tutorials/colors/colors.html
COLOURS = tuple(['r', 'g', 'b', 'orange'])

LabelItem = namedtuple('LabelItem', ['id', 'name'])


class Label:
    """ Holds labels and basic handlers for BACH dataset """
    NORMAL = LabelItem(0, "Normal")
    BENIGN = LabelItem(1, "Benign")
    INSITU = LabelItem(2, "In Situ")
    INVASIVE = LabelItem(3, "Invasive")

    CHOICES = (NORMAL, BENIGN, INSITU, INVASIVE)
    INDEX = dict(CHOICES)

    @classmethod
    def is_valid_option(cls, id_):
        """ Returns true if the id_ belongs to any of the choices """
        return id_ in cls.INDEX.keys()

    @classmethod
    def get_name(cls, id_):
        """ Returns the name associated with the provided label id """
        if not cls.is_valid_option(id_):
            raise LabelIdInvalid()

        return cls.INDEX[id_]

    @classmethod
    def get_choices_as_string(cls):
        """ Returns labels information """
        return ', '.join(tuple('{} : {}'.format(*item) for item in cls.CHOICES))


class PCamLabel:
    """ Holds labels and basic handlers for PCam dataset """

    NORMAL = LabelItem(0, "Normal")
    TUMOR = LabelItem(1, "Tumor")

    CHOICES = (NORMAL, TUMOR)
    INDEX = dict(CHOICES)

    @classmethod
    def is_valid_option(cls, id_):
        """ Returns true if the id_ belongs to any of the choices """
        return id_ in cls.INDEX.keys()

    @classmethod
    def get_name(cls, id_):
        """ Returns the name associated with the provided label id """
        if not cls.is_valid_option(id_):
            raise PCamLabelIdInvalid()

        return cls.INDEX[id_]

    @classmethod
    def get_choices_as_string(cls):
        """ Returns labels information """
        return ', '.join(tuple('{} : {}'.format(*item) for item in cls.CHOICES))


class SubDataset:
    # TODO: replace it with gtorch_utils.constants.DB
    """ Holds basic subdatset names  """
    TRAIN = 'train'
    TEST = 'test'

    SUB_DATASETS = [TRAIN, TEST]


class PCamSubDataset:
    """
    Holds PatchCamelyon subdatset names

    NOTE: we created this class just because PatchCamelyon VALIDATION uses a key
          fifferent than 'val'
    """
    TRAIN = 'train'
    VALIDATION = 'valid'
    TEST = 'test'

    SUB_DATASETS = [TRAIN, VALIDATION, TEST]


class CodeType:
    """ Holds the types of codes available  """
    RAW = LabelItem(0, 'Raw data')
    RANDFACE = LabelItem(1, 'Random-face feature descriptors')
    CNN = LabelItem(2, 'CNN codes')

    CHOICES = (RAW.id, RANDFACE.id, CNN.id)
    TEMPLATES = {
        RAW.id: settings.RAW_CODES_FOLDER,
        RANDFACE.id: settings.RANDOM_FACE_FOLDER,
        CNN.id: settings.CNN_CODES_FOLDER
    }

    @classmethod
    def get_folder(cls, id_):
        """ Returns folder path for the type of code provided """
        assert id_ in cls.CHOICES
        return cls.TEMPLATES[id_]


class ProcessImageOption:
    """ Holds the types of options to process image channels """
    CONCATENATE = LabelItem(0, 'Concatenate')
    GRAYSCALE = LabelItem(1, 'Grayscale')
    MEAN = LabelItem(2, 'Mean')

    CHOICES = (CONCATENATE.id, GRAYSCALE.id, MEAN.id)

    @classmethod
    def is_valid_option(cls, option):
        """ Returns true if the id_ belongs to any of the choices """
        assert isinstance(option, LabelItem)
        return option.id in cls.CHOICES
