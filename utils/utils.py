# -*- coding: utf-8 -*-
""" utils.utils """

import json
import os
import random
import shutil
import string

import numpy as np

import settings


def clean_create_folder(folder_path):
    """
    Removes the folder and recreates it

    Args:
        folder_path (str): path to the folder to be (re)-created
    """
    assert isinstance(folder_path, str)

    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)

    os.mkdir(folder_path)


def get_random_string(length=15):
    """
    Returns a random string with the length especified

    Args:
        length (int): string length

    Returns:
        str
    """
    assert isinstance(length, int)
    assert length > 0

    return ''.join(random.choices(string.ascii_letters+string.digits, k=length))


def get_filename_and_extension(string_):
    """
    Extracts and returns the file name and extension from a file path or file name

    Args:
        string_ (str): File path or file name

    Returns:
        '<filename>', '<extension>'
    """
    assert isinstance(string_, str)
    assert bool(string_), 'Empty strings are not allowed'

    bits = os.path.basename(string_).split('.')

    if len(bits) > 2:
        return ''.join(bits[:-1]), bits[-1]
    if len(bits) == 2:
        return bits[0], bits[1]

    return bits[0], ''


def clean_json_filename(filename):
    """
    Verifies and returns a filename, or random string if not provided, with the .json extension.

    Args:
        filename (str): file name

    Returns
        '<filname or random string>.json'
    """
    assert isinstance(filename, str)

    if filename:
        assert filename.endswith('.json'), 'The filename does not have a .json extension'

    if not filename:
        filename = get_random_string() + '.json'

    return filename


def load_cnn_codes(filename, numpy=True):
    """
    Loads and returns a dictionary with the CNN codes from filename at settings.CNN_CODES_FOLDER.
    Optionaly, cast the lists to numpy arrays

    Args:
        filename (str): Filename with json extension from settings.CNN_CODES_FOLDER
        numpy   (bool): Where to cast the lists to numpy arrays or not.

    Returns:
        {'cnn_codes': <list o lists or numpy array>, 'labels': <list or numpy array>}
    """
    clean_json_filename(filename)

    file_path = os.path.join(settings.CNN_CODES_FOLDER, filename)

    assert os.path.isfile(file_path), '{} does not exit'.format(file_path)

    with open(file_path, 'r') as file_:
        cnn_codes = json.load(file_)

    if numpy:
        for key in cnn_codes:
            cnn_codes[key] = np.array(cnn_codes[key])

    return cnn_codes
