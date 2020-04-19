# -*- coding: utf-8 -*-
""" utils.utils """

import os
import shutil


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
