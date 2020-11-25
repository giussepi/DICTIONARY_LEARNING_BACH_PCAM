# -*- coding: utf-8 -*-
""" utils/datasets/mixins """

import matplotlib.pyplot as plt
from tqdm import tqdm


class CreateJSONFilesMixin:
    """
    Mixins to be used with descendants of utils.datasets.bach.BasePrepareDataset class
    """

    def _create_json_files(self):
        """
        Reads the images from self.image_list and creates the
        the WSI JSON files
        """
        print("Processing images to create whole image JSON files")
        for image_path, folder in tqdm(self.image_list):
            image = plt.imread(image_path)
            h, w = image.shape[:2]

            self._create_image_json_file(
                self._format_clean_filename(image_path, 0, 0),
                folder, image_path, 0, 0, w, h
            )
