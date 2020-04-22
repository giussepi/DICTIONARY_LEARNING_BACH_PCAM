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

### Handle resnet18: fine-tuned / fixed feature extractor
``` python
from dl_models.fine_tuned_resnet_18.models import TransferLearningResnet18

# example 1: Train a resnet18 using fine tuning
model = TransferLearningResnet18(fine_tune=True)
model.training_data_plot_grid()
model.train(num_epochs=25)
model.save('mymodel.pt')
model.visualize_model()
model.test()

# example 2: Load a resnet18 as a fixed feature extractor
model2 = TransferLearningResnet18(fine_tune=False)
model2.load('weights/resnet18_feature_extractor.pt')
model2.visualize_model()
model2.test()
```

### Create datasets for LC_KSVD
```python
from dl_models.fine_tuned_resnet_18.models import TransferLearningResnet18

model = TransferLearningResnet18(fine_tune=True)
model.load('weights/resnet18_fine_tuned.pt')
model.create_datasets_for_LC_KSVD('my_dataset.json')
```
	Note: See function definition to pass the correct parameters

### load_cnn_codes
```python
from utils.utils import load_cnn_codes

test = load_cnn_codes('attempt2_test.json')
test['cnn_codes'].shape  # (512, 2100)
test['labels'].shape  # (4, 2100)
```
	Note: See function definition to pass the correct parameters

## Committing changes made on third-party repositories

   All changes made on any third-party repository from `dl_algorithms` directory must committed and pushed to each repository manually. Because, this application is not keeping track of any of them.
