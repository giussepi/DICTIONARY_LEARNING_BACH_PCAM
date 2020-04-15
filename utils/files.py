# -*- coding: utf-8 -*-
""" utils/files """


def get_name_and_extension(file_name):
    """
    Returns a tuple with the name and extension
    (name, extension)
    """
    assert isinstance(file_name, str)

    bits = file_name.split('.')

    assert len(bits) >= 2

    return '.'.join(bits[:-1]), bits[-1]
