# Dictionary Learning Tests

Testing dictionary learning techniques vs perceptron and MLP using BACH and PatchCamelyon datasets.

## Installation

   1. Create a virtual environment (suggested) using [virtualenv](https://virtualenv.pypa.io/en/stable/) or [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/stable/)

   2. Activate your virtual environment

   3. Install the dependencies

       `pip install -r requirements.txt --use-feature=2020-resolver --no-cache-dir`

   4. [Install the right pytorch version](https://pytorch.org/) for your CUDA vesion. To see your which CUDA version you have just run `nvcc -V`.

   5. Download your datasets:

      1. [ICIAR 2018 Grand Challenge on Breast Cancer Histology images](https://iciar2018-challenge.grand-challenge.org/Dataset/).

      2. [PatchCamelyon (PCam) deep learning classification benchmark](https://github.com/basveeling/pcam)

   5. Copy settings.py.template into settings.py and set the general configuration settings properly

	  `$ cp settings.py.template settings.py`


## BACH_ICIAR_2018

### Transform Dataset

#### Rescale / Resize

``` python
from utils.datasets.bach import RescaleResize

# BACH only
RescaleResize(.0625)()  # rescales using .0625 scaling factor
RescaleResize((100, 100, 3))()  # resizes to (100, 100, 3)
```
	Note: See class definition to pass the correct parameters

**Don't forget to update the path of settings.TRAIN_PHOTOS_DATASET**. E.g.: If you resized to .0625 then
you have to update your settings like this:

``` python
TRAIN_PHOTOS_DATASET = os.path.join(BASE_DATASET_LINK, 'ICIAR2018_BACH_Challenge', 'Photos_0.0625')
```

### Create Train/Validation/Test split
```python
from utils.datasets.bach import TrainValTestSplit

# BACH only
TrainValTestSplit()()
```
	Note: See class definition to pass the correct parameters

### Create ROI files
#### Using whole images
```python
from utils.datasets.bach import WholeImage

# BACH only
WholeImage()()
```
	Note: See class definition to pass the correct parameters

#### Using mini-patches
```python
from utils.datasets.bach import MiniPatch

# BACH only
MiniPatch()()
```
	Note: See class definition to pass the correct parameters

#### Work with a fixed number of mini-patches per image

Use it right after executing `MiniPatch()()`.

``` python
from utils.datasets.bach import SelectNRandomPatches

# BACH only
SelectNRandomPatches(100)()
```

### Plot/Save images from json images
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

# BACH only
# example 1: Train a resnet18 using fine tuning
model = TransferLearningResnet18(fine_tune=True)
model.training_data_plot_grid()
model.train(num_epochs=25)
model.save('fine_tuned_resnet18.pt')
model.visualize_model()
model.test()

# example 2: Load a resnet18 as a fixed feature extractor
# download and save the pre-trained resnet 18
model = TransferLearningResnet18(fine_tune=False)
model.save('resnet18_feature_extractor.pt')
# Load the fixed feature extractor
model2 = TransferLearningResnet18(fine_tune=False)
model2.load('resnet18_feature_extractor.pt')
model2.visualize_model()
model2.test()
```

### Feature extraction / Dimensionality reduction

#### Raw images
If your images are big, you should consider using `RescaleResize` and/or `MiniPatch` classes
to reduce their dimensionality. Thus, you will avoid issues with memory.
``` python
from gtorch_utils.constants import DB
from utils.datasets.bach import RawImages
from constants.constants import ProcessImageOption, Label, PCamLabel, PCamSubDataset

# for Bach
ri = RawImages(process_method=ProcessImageOption.GRAYSCALE, label_class=Label, sub_datasets=DB)
# for PatchCamelyon
ri = RawImages(process_method=ProcessImageOption.GRAYSCALE, label_class=PCamLabel, sub_datasets=PCamSubDataset)

ri.create_datasets_for_LC_KSVD('my_raw_dataset.json')
```
	Note: See function definition to pass the correct parameters

#### Random Faces feature descriptors
``` python
from gtorch_utils.constants import DB
from utils.datasets.bach import RandomFaces
from constants.constants import ProcessImageOption, Label, PCamLabel, PCamSubDataset

# BACH
# Requires all images to have the same width & height so without applying RescaleResize execute the
# TrainValTestSplit()(), then in the settings make sure CUT_SIZE = 512; finally, create minipatches
# MiniPatch()(). Now you'll be able to apply the RandomFaces feature extractor. (of course you can
# change the 512 value, preferably choose a multiple of 32)
randfaces = RandomFaces(img_height=512, img_width=512, process_method=ProcessImageOption.GRAYSCALE, label_class=Label, sub_datasets=DB)
# PatchCamelyon
# if you ran HDF5_2_PNG with only_center=True then the images are 32x32, otherwise they will be 96x96
randfaces = RandomFaces(img_height=32, img_width=32, process_method=ProcessImageOption.GRAYSCALE, label_class=PCamLabel, sub_datasets=PCamSubDataset)

randfaces.create_datasets_for_LC_KSVD('my_randface_dataset.json')
```
	Note: See function definition to pass the correct parameters

#### Sparse codes

This feature extractor requires a learned dictionary learning `D`. Thus, you
should first train your dictionary learning algorithm (e.g.: LC-KSVD1,
LC-KSVD2), save the learned dictionary as a NumPy file `np.save('D.npy', D,
False)`; finally, use the learned dictionary `D` to create the sparse codes.

```python
import numpy as np
from gtorch_utils.constants import DB
from lcksvd.dksvd import DKSVD

from constants.constants import ProcessImageOption, Label, PCamLabel, \
    PCamSubDataset, CodeType
from utils.datasets.bach import SparseCodes
from utils.utils import load_codes


# GETTING LEARNED DICTIONARY ##############################################
test = load_codes('my_raw_dataset_test.json', type_=CodeType.RAW)
train = load_codes('my_raw_dataset_train.json', type_=CodeType.RAW)
val = load_codes('my_raw_dataset_val.json', type_=CodeType.RAW)

SPARSITYTHRES = 15
lcksvd = DKSVD(
    sparsitythres=SPARSITYTHRES, dictsize=train['labels'].shape[0]*SPARSITYTHRES, timeit=True,
    sqrt_alpha=.0012, sqrt_beta=.0012, tol=1e-6, iterations=50, iterations4ini=20
)
Dinit, Tinit_T, Winit_T, Q = lcksvd.initialization4LCKSVD(*train.values())
np.save('Dinit.npy', Dinit, False)
np.save('Tinit_T.npy', Tinit_T, False)
np.save('Winit_T.npy', Winit_T, False)
np.save('Q.npy', Q, False)

D, X, T, W = lcksvd.labelconsistentksvd2(train['codes'], Dinit, train['labels'], Q, Tinit_T, Winit_T)
np.save('D.npy', D, False)
np.save('X.npy', X, False)
np.save('T.npy', T, False)
np.save('W.npy', W, False)
predictions, gamma = lcksvd.classification(D, W, test['codes'])
print('\nFinal recognition rate for LC-KSVD2 is : {0:.4f}'.format(
    accuracy_score(np.argmax(test['labels'], axis=0), predictions)))

# CREATING SPARSE CODES ###################################################
# for BACH
ri = SparseCodes(
    process_method=ProcessImageOption.GRAYSCALE, label_class=Label, sub_datasets=DB,
    sparse_coding=DKSVD.get_sparse_representations,
    sparse_coding_kwargs=dict(D=np.load('D.npy'), sparsitythres=15)
)
# for PatchCamelyon
ri = SparseCodes(
    process_method=ProcessImageOption.GRAYSCALE, label_class=PCamLabel, sub_datasets=PCamSubDataset
    sparse_coding=DKSVD.get_sparse_representations,
    sparse_coding_kwargs=dict(D=np.load('D.npy'), sparsitythres=15)
)

ri.create_datasets_for_LC_KSVD('sparse_codes_dataset.json')
```

#### CNN codes
##### Create datasets for LC_KSVD
```python
from dl_models.fine_tuned_resnet_18.models import TransferLearningResnet18

# BACH only
model = TransferLearningResnet18(fine_tune=True)
model.load('fine_tuned_resnet18.pt')
model.create_datasets_for_LC_KSVD('my_cnn_dataset.json')
```
	Note: See function definition to pass the correct parameters

##### load_codes
```python
from utils.utils import load_codes
from constants.constants import CodeType

# Choose the right code type based on constants.constants.CodeType
test = load_codes(''my_cnn_dataset_test.json', type_=CodeType.CNN)
print(test['codes'].shape)
print(test['labels'].shape)

train = load_codes('my_cnn_dataset_train.json', type_=CodeType.CNN)
print(train['codes'].shape)
print(train['labels'].shape)

val = load_codes('my_cnn_dataset_val.json', type_=CodeType.CNN)
print(val['codes'].shape)
print(val['labels'].shape)
```
	Note: See function definition to pass the correct parameters

### Run LC-KSVD1
```python
import numpy as np
from lc_ksvd.dksvd import DKSVD
from sklearn.metrics import accuracy_score

from constants.constants import CodeType
from utils.utils import load_codes

train = load_codes('my_cnn_dataset_train.json', type_=CodeType.CNN)
val = load_codes('my_cnn_dataset_val.json', type_=CodeType.CNN)
test = load_codes(''my_cnn_dataset_test.json', type_=CodeType.CNN)

lcksvd = DKSVD(dictsize=570, timeit=True)
Dinit, Tinit_T, Winit_T, Q = lcksvd.initialization4LCKSVD(*train.values())
D, X, T, W = lcksvd.labelconsistentksvd1(train['codes'], Dinit, train['labels'], Q, Tinit_T)
predictions, gamma = lcksvd.classification(D, W, test['codes'])
print('\nFinal recognition rate for LC-KSVD1 is : {0:.4f}'.format(
    accuracy_score(np.argmax(test['labels'], axis=0), predictions)))

```
	Note: See function definition to pass the correct parameters

### Run LC-KSVD2
```python
import numpy as np
from lc_ksvd.dksvd import DKSVD
from sklearn.metrics import accuracy_score

from constants.constants import CodeType
from utils.utils import load_cnn_codes

train = load_codes('my_cnn_dataset_train.json', type_=CodeType.CNN)
val = load_codes('my_cnn_dataset_val.json', type_=CodeType.CNN)
test = load_codes(''my_cnn_dataset_test.json', type_=CodeType.CNN)

lcksvd = DKSVD(dictsize=570, timeit=True)
 Dinit, Tinit_T, Winit_T, Q = lcksvd.initialization4LCKSVD(*train.values())

D, X, T, W = lcksvd.labelconsistentksvd2(train['codes'], Dinit, train['labels'], Q, Tinit_T, Winit_T)
predictions, gamma = lcksvd.classification(D, W, test['codes'])
print('\nFinal recognition rate for LC-KSVD2 is : {0:.4f}'.format(
    accuracy_score(np.argmax(test['labels'], axis=0), predictions)))
```
	Note: See function definition to pass the correct parameters

### Run D-KSVD
```python
import numpy as np
from lc_ksvd.dksvd import DKSVD
from sklearn.metrics import accuracy_score

from constants.constants import CodeType
from utils.utils import load_cnn_codes

train = load_codes('my_cnn_dataset_train.json', type_=CodeType.CNN)
val = load_codes('my_cnn_dataset_val.json', type_=CodeType.CNN)
test = load_codes(''my_cnn_dataset_test.json', type_=CodeType.CNN)

lcksvd = DKSVD(dictsize=570, timeit=True)
Dinit, Winit = lcksvd.initialization4DKSVD(*train.values())
predictions, gamma = lcksvd.classification(Dinit, Winit, train['codes'])
print('\nFinal recognition rate for D-KSVD is : {0:.4f}'.format(
    accuracy_score(np.argmax(train['labels'], axis=0), predictions)))
```
	Note: See function definition to pass the correct parameters

### Run Resnet18
```python

from constants.constants import CodeType
from dl_models.fine_tuned_resnet_18.models import TransferLearningResnet18
from utils.datasets.bach import BACHDataset, BachTorchNetDataset

# BACH only
# Train by reading images from disk
model = TransferLearningResnet18(fine_tune=True)
# or train by reading extracted codes (e.g. using raw codes from 64x64 mini-patches)
model = TransferLearningResnet18(
    fine_tune=True, dataset_handler=BachTorchNetDataset,
    dataset_kwargs=dict(
        code_type=CodeType.RAW,
        filename_pattern='my_raw_dataset.json',
        original_shape=(64, 64)
    )
)
#
model.training_data_plot_grid()
model.train(num_epochs=25)
model.save('fine_tuned_resnet18.pt')
model.visualize_model()
model.test()
```
	Note: See function definition to pass the correct parameters

### Visualization tools
#### Visualize learned representations
``` python
import numpy as np
from lc_ksvd.constants import PlotFilter
from lc_ksvd.dksvd import DKSVD
from lc_ksvd.utils.plot_tools import LearnedRepresentationPlotter

from constants.constants import Label, COLOURS, CodeType
from utils.utils import load_cnn_codes

train = load_codes('my_cnn_dataset_train.json', type_=CodeType.CNN)
val = load_codes('my_cnn_dataset_val.json', type_=CodeType.CNN)
test = load_codes(''my_cnn_dataset_test.json', type_=CodeType.CNN)

lcksvd = DKSVD(dictsize=570, timeit=True)
 Dinit, Tinit_T, Winit_T, Q = lcksvd.initialization4LCKSVD(*train.values())

D, X, T, W = lcksvd.labelconsistentksvd2(train['codes'], Dinit, train['labels'], Q, Tinit_T, Winit_T)
predictions, gamma = lcksvd.classification(D, W, test['codes'])

LearnedRepresentationPlotter(predictions=predictions, gamma=gamma, label_index=Label.INDEX, custom_colours=COLOURS)(simple='')

LearnedRepresentationPlotter(predictions=predictions, gamma=gamma, label_index=Label.INDEX, custom_colours=COLOURS)(file_saving_name='myimage')

LearnedRepresentationPlotter(predictions=predictions, gamma=gamma, label_index=Label.INDEX, custom_colours=COLOURS)( filter_by=PlotFilter.UNIQUE, marker='.')

```
	Note: See class definition to pass the correct parameters

#### Visualize dictionary atoms
``` python
from lc_ksvd.dksvd import DKSVD
from lc_ksvd.utils.plot_tools import AtomsPlotter

from constants.constants import Label, COLOURS, CodeType
from utils.utils import load_cnn_codes

train = load_codes('my_cnn_dataset_train.json', type_=CodeType.CNN)
val = load_codes('my_cnn_dataset_val.json', type_=CodeType.CNN)
test = load_codes(''my_cnn_dataset_test.json', type_=CodeType.CNN)

lcksvd = DKSVD(dictsize=570, timeit=True)
 Dinit, Tinit_T, Winit_T, Q = lcksvd.initialization4LCKSVD(*train.values())

D, X, T, W = lcksvd.labelconsistentksvd2(train['codes'], Dinit, train['labels'], Q, Tinit_T, Winit_T)
predictions, gamma = lcksvd.classification(D, W, test['codes'])

AtomsPlotter(dictionary=D, img_width=128, img_height=96, n_rows=10, n_cols=16)()
```
	Note: See class definition to pass the correct parameters


## PatchCamelyon (PCam)

Once you downloaded and updated your settings file properly you have to adapt/format the PCam dataset. Then,
you can use any of tools defined after the BACH sub-section Plot/Save images
from json images (including it).

### Adapt/format dataset

1. HDF5 to PNG

	Update the path of `settings.BASE_DATASET_LINK` before running it.
	Set `settings.TRAIN_PHOTOS_DATASET = os.path.join(BASE_DATASET_LINK, 'images')` before running it.

	``` python
	from utils.datasets.pcam HDF5_2_PNG

	HDF5_2_PNG(only_center=True)()
	```

2. Format split dataset provided by PatchCamelyon

	Set `settings.TRAIN_PHOTOS_DATASET = os.path.join(BASE_DATASET_LINK, 'images')` before running it.

	``` python
	from utils.datasets.pcam FormatProvidedDatasetSplits

	FormatProvidedDatasetSplits()()
	```

3. Create ROI files

   Using whole images
   ```python
   from utils.datasets.pcam import WholeImage

	WholeImage()()
	```
