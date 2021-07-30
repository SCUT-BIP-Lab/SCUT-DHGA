# SCUT-DHGA
This is the official repository of "[Dynamic Hand Gesture Authentication Dataset and Benchmark](https://ieeexplore.ieee.org/abstract/document/9249008)".

## Dataset
We introduce a new dataset **SCUT-DHGA** which is the first large-scale Dynamic-Hand-Gestures authentication dataset. SCUT-DHGA contains 29,160 dynamic-hand-gesture video sequences and more than 1.86 million frames for both color and depth modalities acquired from 193 volunteers. Six kinds of dynamic hand gestures are carefully designed: 1)make a fist starting from thumb, 2)rotate hand while making a fist starting from little finger, 3)catch and then release, 4)four fingers(index, middle, ring, little) touching thumb one by one, 5)bend four fingers(same as 4)) one by one, 6)open a fist starting from thumb.

Now a small part of our dataset containing both depth and color gesture videos from *five* subjects can be downloaded [here](https://drive.google.com/file/d/12vp-6o1gIJLfcw492EJILIXNqxAIcwRU/view?usp=sharing). Note that the depth data is 8-bit in this demo version. The 16-bit depth data will be released soon in our complete dataset.

## How to run

### 1. Prepare the code and python environment
```bash
git clone https://github.com/SCUT-BIP-Lab/SCUT-DHGA.git
cd SCUT-DHGA
pip install -r requirement.txt
```

### 2. Pepare the dataset and place it somewhere

### 3. Just run the code
#### for training
```python
python main.py \
--training_file $path to the training config file$ \
--testing_file $path to the test config file$ \
--data_root $path to the dataset$ \
--train
```

#### for testing
```python
python main.py \
--training_file $path to the training config file$ \
--testing_file $path to the test config file$ \
--data_root $path to the dataset$ \
--test \
--testmodel_name $path to the parameters$
    
```
