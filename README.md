# BACH_ICIAR_2018

Attempt to improve classification results on
[ICIAR 2018 Grand Challenge on Breast Cancer Histology images](https://iciar2018-challenge.grand-challenge.org/Dataset/) by using Dictionary Learning techniques.


## Installation

   1. Create a virtual environment (suggested) using virtualenv or virtualenvwrapper

   2. Activate your virtual environment

   3. Make install.ssh executable

       `$ chmod +x install.ssh`

   4. Install/update libraries and third-party repositories.

       `$ ./install.ssh`

   5. Copy settings.py.template into settings.py and set the general configuration settings properly

	  `$ cp settings.py.template settings.py`


## Usage

### Create Minipatches
```python
from utils.datasets.bach import MiniPatch

MiniPatch(cut_size=608)()
```
	Note: See class definition to pass the correct parameters


### Create Train/Test split
```python
from utils.datasets.bach import TrainTestSplit

TrainTestSplit()()
```
	Note: See class definition to pass the correct parameters


### Plot/save images from json image minipatches
```python
import os

import settings
from utils.datasets.bach import plot_n_first_json_images

    plot_n_first_json_images(5, os.path.join(settings.OUTPUT_FOLDER, settings.TRAIN_FOLDER_NAME),
                             (9, 9), carousel=True)
```
	Note: See function definition to pass the correct parameters


## Committing changes made on third-party repositories

   All changes made on any third-party repository from `dl_algorithms` directory must committed and pushed to each repository manually. Because, this application is not keeping track of any of them.
