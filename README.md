# Semantic Segmentation Project
[Udacity CarND Term 3 Semantic Segmentation Project]
(https://github.com/udacity/CarND-Semantic-Segmentation)

---

## Overview
In this project, the aim is to label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Run
Run the following command to run the project:
```
python main.py
```

### Submission
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder

### Training
Training performed on CPU, for 50 epochs, the BATCH_SIZE is 8, the learning rate is 0.001, and the keep prob is 0.5.
Logs for epochs from 37 to 50 are as follows
loss calculated by summing the loss values of batches
iou calculated by taking the weighted average of iou values of batches
ET is the epoch time
TT is the total time from the beginning.
| Epoch |   loss   |   iou   |    ET    |    TT    |
|:-----:|:--------:|:-------:|:--------:|:--------:|
|  37   | 4.255037 |0.068525 |  1085.30 | 40340.56 |
|  38   | 4.056522 |0.067237 |  1077.36 | 41421.84 |
|  39   | 3.948550 |0.065984 |  1087.49 | 42513.20 |
|  40   | 4.221526 |0.064803 |  1085.30 | 43602.50 |
|  41   | 3.667796 |0.063658 |  1086.02 | 44692.32 |
|  42   | 3.469842 |0.062540 |  1086.04 | 45782.36 |
|  43   | 3.444155 |0.061450 |  1085.57 | 46871.77 |
|  44   | 3.210744 |0.060396 |  1085.98 | 47961.74 |
|  45   | 3.458068 |0.059383 |  1074.84 | 49040.52 |
|  46   | 3.393176 |0.058424 |  1085.10 | 50129.42 |
|  47   | 2.931456 |0.057482 |  1054.72 | 51188.06 |
|  48   | 2.752962 |0.056551 |  1056.54 | 52248.63 |
|  49   | 2.564550 |0.055631 |  1068.14 | 53320.68 |
|  50   | 2.399701 |0.054733 |  1087.81 | 54412.21 |