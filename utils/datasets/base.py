# -*- coding: utf-8 -*-
""" utils/datasets/base """

import torch
from gutils.numpy_.numpy_ import LabelMatrixManager
from gtorch_utils.datasets.generic import BaseDataset

from utils.utils import load_codes, clean_json_filename, \
    get_filename_and_extension


class BaseTorchDataset(BaseDataset):
    """
    Base torch dataset handler

    NOTE: not to be used directly, its descendants must be used instead
    """

    def __init__(self, subset, sub_datasets, **kwargs):
        """
        Loads the subdataset

        Args:
           filename_pattern (str): filename with .json extension used to create the codes
                                   when the calling the create_datasets_for_LC_KSVD method.
           code_type   (CodeType): Code type used. See constants.constants.CodeType class defition
           sub_datasets ():
        """
        assert subset in sub_datasets.SUB_DATASETS
        self.subset = subset
        filename_pattern = kwargs.get('filename_pattern')
        assert isinstance(filename_pattern, str)

        code_type = kwargs.get('code_type')
        cleaned_filename = clean_json_filename(filename_pattern)
        name, extension = get_filename_and_extension(cleaned_filename)
        file_name = '{}_{}.{}'.format(name, subset, extension)
        self.data = load_codes(file_name, type_=code_type)
        self.data['labels'] = LabelMatrixManager.get_1d_array_from_2d_matrix(self.data['labels'])

    def __len__(self):
        """
        Returns:
            dataset size (int)
        """
        return self.data['labels'].shape[0]

    def __getitem__(self, idx):
        """
        Returns:
            dict(feats=..., label=...)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return dict(
            feats=torch.from_numpy(self.data['codes'][:, idx].ravel()).float(),
            label=self.data['labels'][idx]
        )
