# -*- coding: utf-8 -*-
""" core/exceptions/dataset """


class ImageNameInvalid(Exception):
    """
    Exception to be raised when the filename does not refers to any of the defined labels
    """

    def __init__(self, message=''):
        """  """
        # Avoiding circular dependency
        from constants.constants import Label  # NOQA
        if not message:
            message = 'Filename does not refer to any of the four defined classes: {}'.format(
                Label.get_choices_as_string())
        super().__init__(message)


class LabelIdInvalid(Exception):
    """
    Exception to be raised when an id provided does not belong to any of
    the BACH labels
    """

    def __init__(self, message=''):
        """  """
        # Avoiding circular dependency
        from constants.constants import Label  # NOQA
        if not message:
            message = 'The id provided is not a between the valid options: {}'.format(
                Label.get_choices_as_string())
        super().__init__(message)


class PCamLabelIdInvalid(Exception):
    """
    Exception to be raised when an id provided does not belong to any of
    the PCam labels
    """

    def __init__(self, message=''):
        """  """
        # Avoiding circular dependency
        from constants.constants import PCamLabel  # NOQA
        if not message:
            message = 'The id provided is not a between the valid options: {}'.format(
                PCamLabel.get_choices_as_string())
        super().__init__(message)
