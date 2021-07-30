import os, csv
import time
import shutil
import argparse
import random
import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
from model import i3d_auth, i3d_auth_RGBD
from Datasets import RGB_dataset
from utils import CenterLoss, calculate_eer
from utils.transforms import Random_gamma
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class Instructor:
    def __init__(self, args):
        self.args = args
        '''
        dataset and dataloader
        '''
        transform = Random_gamma()
        dataset = RGB_dataset(self.args, transforms=transform)

        if(self.args.train):
            self.args.num_classes = dataset.get_classes()
            self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        else:
            self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        '''
        net and load state_dict
        '''
        self.net = args.net(args)
        if(self.args.train):
            pretrained_path = os.path.join('work_dir', 'state_dict', 'pretrained', self.args.pretrained_name + '.pt')
            load_dict = torch.load(pretrained_path)
        else:
            test_path = self.args.testmodel_name
            load_dict = torch.load(test_path)
        net_state_dict = self.net.state_dict()
        load_dict = {k: v for k, v in load_dict.items() if k in net_state_dict}
        net_state_dict.update(load_dict)
        self.net.load_state_dict(net_state_dict)
        self._print_args()
        self.net.cuda()

        '''
        loss func and optimizer
        '''
        if(self.args.train):
            if self.args.feature_mode == 'linear':
                self.id_center_loss = CenterLoss(self.args.num_classes, 128, 1).cuda()
            elif self.args.feature_mode == 'time_distrubuted':
                self.id_center_loss = CenterLoss(self.args.num_classes, 128, 1).cuda()
            params = list(self.net.parameters()) + list(self.id_center_loss.parameters())
            self.optimizer = optim.SGD(params, lr=self.args.init_lr, momentum=0.9, weight_decay=0.0000001)
            self.lr_sched = optim.lr_scheduler.MultiStepLR(self.optimizer, [10, 15])

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.net.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print(
            '>> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('>> training arguments:')
        for arg in vars(self.args):
            print('>>> {0}: {1}'.format(arg, getattr(self.args, arg)))

    def train(self):
        log_dir = os.path.join('work_dir', 'log', self.args.exp_name, '{}_{}_lr{}_ctl{}_pre{}'.format(self.args.model_name, self.args.feature_mode, self.args.init_lr, self.args.center_loss_ratio, self.args.pretrained_name))
        if os.path.exists(log_dir):shutil.rmtree(log_dir)
        summary_writer = SummaryWriter(comment='i3d_auth', log_dir=log_dir)
        global_step = 0
        total_loss = tot_center_loss = tot_cls_loss = total_right = 0
        for epoch in range(1, self.args.num_epochs+1):
            self.lr_sched.step()
            print('>' * 100)
            print('epoch:{}/{}'.format(epoch, self.args.num_epochs))
            s_time = time.time()

            for vid, label in tqdm(self.dataloader):
                vid = vid.cuda()
                feature, logits = self.net(vid)
                global_step += 1

                if self.args.feature_mode == 'linear':
                    #format label
                    label = torch.tensor(label).cuda()

                    # loss
                    cls_loss = F.cross_entropy(logits, label)
                    center_loss = self.id_center_loss(label, feature)
                    loss = cls_loss + self.args.center_loss_ratio * center_loss

                    #training acc
                    _, pre = torch.max(logits, 1, True)
                    pre = pre.view(self.args.batch_size)
                    right_samples = torch.sum(label == pre).sum().float()
                    total_right = total_right + right_samples


                elif self.args.feature_mode == 'time_distrubuted':
                    # format label
                    label_tmp = []
                    for label_ in label:
                        l_ = np.zeros((self.args.num_classes, 8), np.float32)
                        for fr in range(8):
                            l_[label_, fr] = 1  # binary classification
                        label_tmp.append(l_)
                    label = torch.tensor(np.asarray(label_tmp)).cuda()

                    # loss
                    feature_dim = feature.shape[1]
                    feature = feature.permute(0, 2, 1).contiguous()
                    feature = feature.view(-1, feature_dim, 1).squeeze()
                    _, label_index = torch.max(label, 1, True)
                    label_index = label_index.permute(0, 2, 1).contiguous()
                    label_index = label_index.view(-1, 1, 1).squeeze(2)
                    center_loss = self.id_center_loss(label_index.squeeze(-1), feature)
                    cls_loss = F.binary_cross_entropy_with_logits(torch.max(logits, dim=2)[0], torch.max(label, dim=2)[0]) + F.binary_cross_entropy_with_logits(logits, label)
                    loss = cls_loss + self.args.center_loss_ratio * center_loss

                    # cal training acc
                    _, pre_index = torch.max(logits, 1, True)
                    _, label_index = torch.max(label, 1, True)
                    right_samples = torch.sum(label_index == pre_index).sum().float()
                    total_right = total_right + right_samples

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                tot_cls_loss += cls_loss.item()
                tot_center_loss += center_loss.item()
                total_loss += loss.item()

                if (global_step % self.args.log_intervals) == 0:
                    summary_writer.add_scalar('tot_loss', total_loss / self.args.log_intervals, global_step)
                    summary_writer.add_scalar('center_loss', tot_center_loss / self.args.log_intervals, global_step)
                    summary_writer.add_scalar('cls_loss', tot_cls_loss / self.args.log_intervals, global_step)
                    if self.args.feature_mode == 'linear':
                        summary_writer.add_scalar('train_acc', total_right / (self.args.log_intervals * self.args.batch_size), global_step)
                    else:
                        summary_writer.add_scalar('train_acc', total_right / (self.args.log_intervals * self.args.batch_size * 8), global_step)

                    print('\nstep:{:.0f}    cls_loss:{:.2f}    center_loss:{:.2f}'.format(global_step, tot_cls_loss / self.args.log_intervals, tot_center_loss / self.args.log_intervals))
                    total_loss = tot_center_loss = tot_cls_loss = total_right = 0
                # if (global_step % (self.args.log_intervals*10)) == 0:
                #     for name, param in self.net.named_parameters():
                #         summary_writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step)

            state_dict = {'net': self.net.state_dict(), 'optimizer': self.optimizer.state_dict(), 'steps': global_step}
            save_path = os.path.join('work_dir', 'state_dict', self.args.exp_name, '{}_{}_lr{}_ctl{}_pre{}'.format(self.args.model_name, self.args.feature_mode, self.args.init_lr, self.args.center_loss_ratio, self.args.pretrained_name))
            if not os.path.exists(save_path): os.makedirs(save_path)
            save_model_name = 'epoch{}.pt'.format(str(epoch))
            torch.save(state_dict, os.path.join(save_path, save_model_name))
            print('epoch_time is {}'.format(time.time()-s_time))

    def test(self):
        self.net.eval()
        vid_names = []
        all_features = []
        for batch, (vids, labels) in enumerate(self.dataloader):
            vids = vids.cuda()
            with torch.no_grad():
                features = self.net(vids)
            all_features.extend(list(features.cpu().numpy()))
            vid_names.extend([str(label) for label in labels])
            print('Finish calculating batch: {}'.format(batch+1))
        
        assert(len(all_features) == len(vid_names))
        names_features = dict(zip(vid_names, all_features))

        fin = open(os.path.join('work_dir', 'config', args.testing_file))
        fin_csv = csv.reader(fin)

        print('Begin calculating eer...')
        features1 = []
        features2 = []
        pair_labels = []
        for i, row in enumerate(fin_csv):
            if(i > 0):
                features1.append(names_features[row[0]])
                features2.append(names_features[row[1]])
                pair_labels.append(row[2] == '1')
        
        eer, threshold = calculate_eer(np.asarray(features1), np.asarray(features2), np.asarray(pair_labels), useCosin=True)
        eer2, threshold2 = calculate_eer(np.asarray(features1), np.asarray(features2), np.asarray(pair_labels), useCosin=False)
        print('Cosin : \n   EER: {}  Threshold: {}'.format(eer, threshold))
        print('Euclidean : \n   EER: {}  Threshold: {}'.format(eer2, threshold2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 实验模式-----------------------------------------------------------------------------------------
    parser.add_argument('--training_file', type=str, default='0_auth1_ges0_train.csv', help='training_file')
    parser.add_argument('--testing_file', type=str, default='6_auth1_all_test.csv', help='testing_file')
    parser.add_argument('--data_root', type=str, default='/home/data/DHG-Auth/color_hand',help='Dataset directory')
    parser.add_argument('--train', dest='train', help='train mode', action='store_true')
    parser.add_argument('--test', dest='train', help='test mode', action='store_false')
    # 训练时输入的参数----------------------------------------------------------------------------------
    parser.add_argument('--exp_name', type=str, default='0_auth1_ges0_train', help='experiments name')
    parser.add_argument('--model_name', type=str, default='i3dauthRGB', help='i3dauthRGB, i3dauthRGBD')
    parser.add_argument('--feature_mode', type=str, default='linear', help='linear or time_distrubuted')
    parser.add_argument('--init_lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='epochs')
    parser.add_argument('--center_loss_ratio', type=float, default=0.001, help='center_loss_ratio')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--log_intervals', type=int, default=20, help='log_intervals')
    parser.add_argument('--pretrained_name', type=str, default='rgb_imagenet', help='rgb_imagenet or rgb_charades')
    parser.add_argument('--testmodel_name', type=str, default='./work_dir/state_dict/0_auth1_ges0_train/i3dauthRGB_linear_lr0.1_ctl0.0_prergb_imagenet/epoch20.pt', help='rgb_imagenet or rgb_charades')
    # 其他
    parser.add_argument('--seed', default=42, type=int)


    args = parser.parse_args()

    model_classes = {
        'i3dauthRGB': i3d_auth,
        'i3dauthRGBD': i3d_auth_RGBD,
        'i3dnonlocal': None
    }

    args.net = model_classes[args.model_name]

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    ins = Instructor(args)
    if(args.train):
        ins.train()
    else:
        ins.test()