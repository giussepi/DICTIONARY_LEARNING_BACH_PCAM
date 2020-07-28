## Usage

### Adapt/format dataset

#### HDF5 to PNG

Update the path of `settings.BASE_DATASET_LINK` before running it.
Set `settings.TRAIN_PHOTOS_DATASET = os.path.join(BASE_DATASET_LINK, 'images')` before running it.

``` python
from utils.datasets.pcam HDF5_2_PNG

HDF5_2_PNG()()
```


#### Format dataset split provided by PatchCamelyon

Set `settings.TRAIN_PHOTOS_DATASET = os.path.join(BASE_DATASET_LINK, 'images')` before running it.

``` python
from utils.datasets.pcam FormatProvidedDatasetSplits

FormatProvidedDatasetSplits()()
```


### Create ROI files
#### Using whole images
```python
from utils.datasets.pcam import WholeImage

WholeImage()()
```
