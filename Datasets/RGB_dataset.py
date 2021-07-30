import torch
import torch.utils.data as data_utl
import numpy as np
import csv
import os
import os.path
import cv2
from collections import Counter

def make_train_dataset(train_csv, dataset_root):
    '''
    :param train_csv: name of training.csv
    :param dataset_root: dataset path
    :return: num_classes and dataset
    '''
    # train mode
    dataset = []
    label_list = []
    fin = open(os.path.join('work_dir', 'config', train_csv))
    fin_csv = csv.reader(fin)
    for i, row in enumerate(fin_csv):
        if i == 0:
            continue
        if row[2] == 'True':
            vid_name = row[0]
            if not os.path.exists(os.path.join(dataset_root, vid_name)):
                print('{} not exists'.format(vid_name))
                continue
            label = int(row[1])
            label_list.append(label)
            dataset.append((os.path.join(dataset_root, vid_name), label))
    assert len(Counter(label_list)) == max(label_list)+1
    return max(label_list)+1, dataset

def make_test_dataset(test_csv, dataset_root):
    # test mode
    # read csv
    fin = open(os.path.join('work_dir', 'config', test_csv))
    fin_csv = csv.reader(fin)

    # fill vid name set
    dataset = set()
    for i, row in enumerate(fin_csv):
        if(i > 0):
            dataset.update(row[:2])

    # (vid_path, vid_name)
    dataset = [(os.path.join(dataset_root, vid_name), vid_name) for vid_name in dataset]
    return dataset


def load_frames(path, nf=64):
    if nf != 64:
        print("error")
    else:
        frames = []
        for i in range(1, nf + 1):
            img = cv2.imread(os.path.join(path, str(i).zfill(2) + '.jpg'))
            frames.append(img)
    
    return np.asarray(frames, dtype=np.float32)


def video_to_tensor(pic):
    pic = pic.astype(np.float32)
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))

class RGB_dataset(data_utl.Dataset):
    def __init__(self, args, transforms=None):
        self.feature_mode = args.feature_mode
        self.isTrain = args.train
        if(self.isTrain):
            self.classes, self.data = make_train_dataset(args.training_file, args.data_root)
        else:
            self.data = make_test_dataset(args.testing_file, args.data_root)
        self.transforms = transforms

    def __getitem__(self, index):
        vid_path, label = self.data[index]
        vid = load_frames(vid_path)
        if self.isTrain and self.transforms is not None:
            vid = self.transforms(vid)

        if not self.isTrain:
            return video_to_tensor(vid), label
        else:
            return video_to_tensor(vid), torch.tensor(label)

    def __len__(self):
        return len(self.data)

    def get_classes(self):
        return self.classes
