
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.append("..")
import random
import numpy as np
import cv2
import torch
from tqdm import tqdm
import argparse
from tensorboardX import SummaryWriter
import csv
import torch.utils.data as data_utl
from model import i3d_auth, i3d_auth_RGBD

def make_dataset(visualize_csv, dataset_root):
	dataset = []
	fin = open(os.path.join('..' ,'work_dir', 'config', visualize_csv))
	fin_csv = csv.reader(fin)
	for i, row in enumerate(fin_csv):
		if i == 0:
			continue
		vid_name = row[0]
		_, _, ID, ges, _ = vid_name.split('_')
		dataset.append((os.path.join(dataset_root, vid_name), ID))
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
		self.data = make_dataset(args.visualize_csv_path, args.data_root)
		self.transforms = transforms

	def __getitem__(self, index):
		vid_path, label = self.data[index]
		vid = load_frames('/home/data/BIP-LAB_gesture_dataset/color_hand/04_05_06')
		if self.transforms is not None:
			vid = self.transforms(vid)
		label = random.randint(1,10)
		return video_to_tensor(vid), torch.tensor(int(label))

	def __len__(self):
		return len(self.data)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--visualize_csv_path', type=str, default='visualize1.csv', help='training_file')
	parser.add_argument('--model_name', type=str, default='i3dauthRGB', help='i3dauthRGB, i3dauthRGBD')
	parser.add_argument('--data_root', type=str, default='/home/data/DHG_Auth/RGB', help='Dataset path')
	parser.add_argument('--feature_mode', type=str, default='linear', help='linear or time_distrubuted')
	parser.add_argument('--batch_size', type=int, default=8, help='batch size')
	parser.add_argument('--description', type=str, default='test', help='description')
	parser.add_argument('--parameter_path', type=str, default='rgb_imagenet')
	args = parser.parse_args()



	model_classes = {
		'i3dauthRGB': i3d_auth,
		'i3dauthRGBD': i3d_auth_RGBD,
		'i3dnonlocal': None
	}

	args.net = model_classes[args.model_name]
	args.train = False

	dataset = RGB_dataset(args)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
	                                         pin_memory=True, drop_last=False)

	net = args.net(args)

	pretrained_dict = torch.load(os.path.join('..' ,'work_dir', 'state_dict', 'pretrained', 'rgb_imagenet' + '.pt'))
	net_state_dict = net.state_dict()
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
	net_state_dict.update(pretrained_dict)
	net.load_state_dict(net_state_dict)
	net.cuda()

	for p in net.parameters():
		p.requires_grad = False
	net.eval()


	flag = False
	log_dir = os.path.join('..' ,'work_dir', 'log', 'visualize''{}'.format(args.model_name, args.feature_mode, args.description))
	summary_writer = SummaryWriter(comment='i3d_auth', log_dir=log_dir)
	for vid, label in tqdm(dataloader):
		vid = torch.randn(8, 3, 64, 200, 200, dtype=torch.float32)
		vid = vid.cuda()
		feature = net(vid)
		if flag == False:
			feature_list = feature
			label_list = label
			flag = True
			continue
		else:
			feature_list = torch.cat([feature_list, feature])
			label_list = torch.cat([label_list, label])

		print(feature_list.shape)
		print(label_list.shape)
	summary_writer.add_embedding(feature_list, metadata=label_list)
