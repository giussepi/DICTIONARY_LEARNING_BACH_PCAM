# BACH_ICIAR_2018

Attempt to improve classification results on
[ICIAR 2018 Grand Challenge on Breast Cancer Histology images](https://iciar2018-challenge.grand-challenge.org/Dataset/) by using Dictionary Learning techniques.


## Installation

   1. Create a virtual environment (suggested) using virtualenv or virtualenvwrapper

   2. Activate your virtual environment

   3. Install the dependencies

       ` pip install -r requirements.txt --use-feature=2020-resolver --no-cache-dir`

   4. Copy settings.py.template into settings.py and set the general configuration settings properly

	  `$ cp settings.py.template settings.py`


## Usage

### Transform Dataset

#### Rescale / Resize
``` python
from utils.datasets.bach import RescaleResize

RescaleResize(.25)()  # rescales using .25 scaling factor
RescaleResize((100, 100, 3))()  # resizes to (100, 100, 3)
```
	Note: See class definition to pass the correct parameters

Don't forget to update the path of `settings.TRAIN_PHOTOS_DATASET`

### Create Train/Test split
```python
from utils.datasets.bach import TrainTestSplit

TrainTestSplit()()
```
	Note: See class definition to pass the correct parameters

### Create ROI files
#### Using whole images
```python
from utils.datasets.bach import WholeImage

WholeImage()()
```
	Note: See class definition to pass the correct parameters

#### Using mini-patches
```python
from utils.datasets.bach import MiniPatch

MiniPatch()()
```
	Note: See class definition to pass the correct parameters

### Plot/Save images from json image minipatches
```python
import os

import settings
from utils.datasets.bach import plot_n_first_json_images

    plot_n_first_json_images(5, os.path.join(settings.OUTPUT_FOLDER, settings.TRAIN_FOLDER_NAME),
                             (9, 9), carousel=True, remove_axes=False, dpi=100)
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

### Feature extraction / Dimensionality reduction

#### Raw images
If your images are big, you should consider using `RescaleResize` and/or `MiniPatch` classes
to reduce their dimensionality. Thus, you will avoid issues with memory.
``` python
from utils.datasets.bach import RawImages
from constants.constants import ProcessImageOption, Label

ri = RawImages(process_method=ProcessImageOption.MEAN, label_class=Label)
data = ri.create_datasets_for_LC_KSVD('my_raw_dataset.json')
```
	Note: See function definition to pass the correct parameters

#### Random Faces feature descriptors
``` python
from utils.datasets.bach import RandomFaces
from constants.constants import ProcessImageOption, Label

randfaces = RandomFaces(img_height=512, img_width=512, process_method=ProcessImageOption.CONCATENATE, label_class=Label)
data = randfaces.create_datasets_for_LC_KSVD('my_raw_dataset.json')
```
	Note: See function definition to pass the correct parameters

#### CNN codes
##### Create datasets for LC_KSVD
```python
from dl_models.fine_tuned_resnet_18.models import TransferLearningResnet18

model = TransferLearningResnet18(fine_tune=True)
model.load('weights/resnet18_fine_tuned.pt')
model.create_datasets_for_LC_KSVD('mydataset.json')
```
	Note: See function definition to pass the correct parameters

##### load_codes
```python
from utils.utils import load_codes
from constants.constants import CodeType

# Choose the right code type based on constants.constants.CodeType
test = load_codes('mydataset_test.json', type_=CodeType.CNN)
test['codes'].shape  # (512, 2100)
test['labels'].shape  # (4, 2100)

train = load_codes('attempt2_train.json', type_=CodeType.CNN)
train['codes'].shape  # (512, 11900)
train['labels'].shape  # (4, 11900)
```
	Note: See function definition to pass the correct parameters

### Run LC-KSVD1
```python
import numpy as np
from sklearn.metrics import accuracy_score

from lc_ksvd.dksvd import DKSVD
from utils.utils import load_cnn_codes


train = load_cnn_codes('attempt3_train.json')
test = load_cnn_codes('attempt3_test.json')

lcksvd = DKSVD(dictsize=570, timeit=True)
Dinit, Tinit_T, Winit_T, Q = lcksvd.initialization4LCKSVD(*train.values())
D, X, T, W = lcksvd.labelconsistentksvd1(train['cnn_codes'], Dinit, train['labels'], Q, Tinit_T)
predictions, gamma = lcksvd.classification(D, W, test['cnn_codes'])
print('\nFinal recognition rate for LC-KSVD1 is : {0:.4f}'.format(
    accuracy_score(np.argmax(test['labels'], axis=0), predictions)))

```
	Note: See function definition to pass the correct parameters

### Run LC-KSVD2
```python
import numpy as np
from sklearn.metrics import accuracy_score

from lc_ksvd.dksvd import DKSVD
from utils.utils import load_cnn_codes


train = load_cnn_codes('attempt3_train.json')
test = load_cnn_codes('attempt3_test.json')

lcksvd = DKSVD(dictsize=570, timeit=True)
 Dinit, Tinit_T, Winit_T, Q = lcksvd.initialization4LCKSVD(*train.values())

D, X, T, W = lcksvd.labelconsistentksvd2(train['cnn_codes'], Dinit, train['labels'], Q, Tinit_T, Winit_T)
predictions, gamma = lcksvd.classification(D, W, test['cnn_codes'])
print('\nFinal recognition rate for LC-KSVD2 is : {0:.4f}'.format(
    accuracy_score(np.argmax(test['labels'], axis=0), predictions)))
```
	Note: See function definition to pass the correct parameters

### Run D-KSVD
```python
import numpy as np
from sklearn.metrics import accuracy_score

from lc_ksvd.dksvd import DKSVD
from utils.utils import load_cnn_codes

train = load_cnn_codes('attempt3_train.json')
test = load_cnn_codes('attempt3_test.json')

lcksvd = DKSVD(dictsize=570, timeit=True)
Dinit, Winit = lcksvd.initialization4DKSVD(*train.values())
predictions, gamma = lcksvd.classification(Dinit, Winit, train['cnn_codes'])
print('\nFinal recognition rate for D-KSVD is : {0:.4f}'.format(
    accuracy_score(np.argmax(train['labels'], axis=0), predictions)))
```
	Note: See function definition to pass the correct parameters

### Visualization tools
#### Visualize learned representations
``` python
import numpy as np

from constants.constants import Label, COLOURS
from lc_ksvd.constants import PlotFilter
from lc_ksvd.dksvd import DKSVD
from lc_ksvd.utils.plot_tools import LearnedRepresentationPlotter
from utils.utils import load_cnn_codes


train = load_cnn_codes('attempt3_train.json')
test = load_cnn_codes('attempt3_test.json')

lcksvd = DKSVD(dictsize=570, timeit=True)
 Dinit, Tinit_T, Winit_T, Q = lcksvd.initialization4LCKSVD(*train.values())

D, X, T, W = lcksvd.labelconsistentksvd2(train['cnn_codes'], Dinit, train['labels'], Q, Tinit_T, Winit_T)
predictions, gamma = lcksvd.classification(D, W, test['cnn_codes'])

LearnedRepresentationPlotter(predictions=predictions, gamma=gamma, label_index=Label.INDEX, custom_colours=COLOURS)(simple='')

LearnedRepresentationPlotter(predictions=predictions, gamma=gamma, label_index=Label.INDEX, custom_colours=COLOURS)(file_saving_name='myimage')

LearnedRepresentationPlotter(predictions=predictions, gamma=gamma, label_index=Label.INDEX, custom_colours=COLOURS)( filter_by=PlotFilter.UNIQUE, marker='.')

```
	Note: See class definition to pass the correct parameters

#### Visualize dictionary atoms
``` python
from lc_ksvd.dksvd import DKSVD
from lc_ksvd.utils.plot_tools import AtomsPlotter
from utils.utils import load_cnn_codes


train = load_cnn_codes('attempt3_train.json')
test = load_cnn_codes('attempt3_test.json')

lcksvd = DKSVD(dictsize=570, timeit=True)
 Dinit, Tinit_T, Winit_T, Q = lcksvd.initialization4LCKSVD(*train.values())

D, X, T, W = lcksvd.labelconsistentksvd2(train['cnn_codes'], Dinit, train['labels'], Q, Tinit_T, Winit_T)
predictions, gamma = lcksvd.classification(D, W, test['cnn_codes'])

AtomsPlotter(dictionary=D, img_width=128, img_height=96, n_rows=10, n_cols=16)()
```
	Note: See class definition to pass the correct parameters


## Committing changes made on third-party repositories

   All changes made on any third-party repository from `dl_algorithms` directory must committed and pushed to each repository manually. Because, this application is not keeping track of any of them.
