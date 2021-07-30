import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        (batch, channel, t, h, w) = x.size()
        # out_t = np.ceil(float(t) / float(self.stride[0]))
        # out_h = np.ceil(float(h) / float(self.stride[1]))
        # out_w = np.ceil(float(w) / float(self.stride[2]))
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.01, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        # print ('x shapes is {}'.format(x.shape))
        (batch, channel, t, h, w) = x.size()
        # print (t, h, w)
        # out_t = np.ceil(float(t) / float(self._stride[0]))
        # out_h = np.ceil(float(h) / float(self._stride[1]))
        # out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                        stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class i3d_auth(nn.Module):
    def __init__(self, args):
        super(i3d_auth, self).__init__()
        self.train_ = args.train
        if self.train_:
            self._id_classes = args.num_classes
        self.feature_mode = args.feature_mode
        self.name = 'i3d_auth'
        self.in_channels = 3

        self.end_points = {}
        self.end_points['Conv3d_1a_7x7'] = Unit3D(in_channels=self.in_channels, output_channels=64, kernel_shape=[7, 7, 7],stride=(2, 2, 2), padding=(3, 3, 3), name=self.name + 'Conv3d_1a_7x7')
        self.end_points['MaxPool3d_2a_3x3'] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        self.end_points['Conv3d_2b_1x1'] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0, name=self.name + 'Conv3d_2b_1x1')
        self.end_points['Conv3d_2c_3x3'] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1, name=self.name + 'Conv3d_2c_3x3')
        self.end_points['MaxPool3d_3a_3x3'] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        self.end_points['Mixed_3b'] = InceptionModule(192, [64, 96, 128, 16, 32, 32], self.name + 'Mixed_3b')
        self.end_points['Mixed_3c'] = InceptionModule(256, [128, 128, 192, 32, 96, 64], self.name + 'Mixed_3c')
        self.end_points['MaxPool3d_4a_3x3'] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)
        self.end_points['Mixed_4b'] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], self.name + 'Mixed_4b')
        self.end_points['Mixed_4c'] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], self.name + 'Mixed_4c')
        self.end_points['Mixed_4d'] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], self.name + 'Mixed_4d')
        self.end_points['Mixed_4e_new'] = InceptionModule(128 + 256 + 64 + 64, [32, 16, 32, 16, 32, 32], self.name + 'Mixed_4e')
        self.end_points['MaxPool3d_5a_2x2'] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)

        if self.feature_mode == 'time_distrubuted':
            self.pool = nn.AdaptiveAvgPool3d(output_size=[8, 1, 1])
            # self.avg_pool = nn.AvgPool3d(kernel_size=[1, 7, 7], stride=(1, 1, 1))
            if self.train_:
                self.id_logits = Unit3D(in_channels=128, output_channels=self._id_classes, kernel_shape=[1, 1, 1], padding=0, activation_fn=None, use_batch_norm=False, use_bias=True, name='logits')


        if self.feature_mode == 'linear':
            self.pool = nn.AdaptiveAvgPool3d(output_size=[1, 1, 1])
            # self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(2, 1, 1))
            if self.train_:
                self.id_logits = nn.Linear(128, self._id_classes)

        self.dropout = nn.Dropout(args.dropout)
        self.build()

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        for end_point in self.end_points:
            x = self._modules[end_point](x)  # use _modules to work with dataparallel

        if self.feature_mode == "time_distrubuted":
            x = self.pool(x)
            feature = x.squeeze(3).squeeze(3)
            if self.train_:
                id_logits = self.id_logits(x).squeeze(3).squeeze(3)
        elif self.feature_mode == "linear":
            x = self.pool(x)
            x = x.squeeze(3).squeeze(3)
            feature = torch.flatten(x, start_dim=1)
            if self.train_:
                id_logits = self.id_logits(feature)

        if self.train_:
            return feature, id_logits
        else:
            return feature

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    # import sys
    # sys.path.append("..")
    # from utils.CenterLoss import CenterLoss
    # from torchsummary import summary

    # i3d = I3d_auth(id_classes=33, feature_mode="time_distrubuted")
    # i3d = i3d_auth(id_classes=33, feature_mode="Linear")
    # i3d = i3d.cuda()
    #
    # summary(i3d, input_size=(3, 64, 224, 224), batch_size=1, device='cuda')
    #
    # dummy_input = torch.FloatTensor(1, 3, 64, 224, 224).cuda()
    # feature, logits = i3d(dummy_input)
    # print(feature.shape)
    # print(logits.shape)

    # id_center_loss = CenterLoss(33, 128 , 1).cuda()
    # for i in range(100):
    #     dummy_input = torch.FloatTensor(1, 3, 64, 200, 200).cuda()
    #     labels_ID = torch.ones(1, 33, 8).cuda()
    #
    #     feature, logits = i3d(dummy_input)
    #
    #     print("feature shape is {}".format(feature.shape))
    #     print("loigts shape is {}".format(logits.shape))
    #
    #     feature_dim = feature.shape[1]
    #     feature = feature.permute(0, 2, 1).contiguous()
    #     feature = feature.view(-1, feature_dim, 1).squeeze()
    #     print("feature shape is {}".format(feature.shape))
    #
    #     _, label_index = torch.max(labels_ID, 1, True)
    #     label_index = label_index.permute(0, 2, 1).contiguous()
    #     label_index = label_index.view(-1, 1, 1).squeeze(2)
    #     print("label index shape is {}".format(label_index.shape))
    #
    #     my_id_center_loss = id_center_loss(label_index.squeeze(-1), feature)
    #     print("my_id_center_loss is {}".format(my_id_center_loss))

    # summary(i3d, input_size=(3, 64, 200, 200), batch_size=1, device='cuda')

    # label = np.zeros((33, 8), np.float32)
    # for ann in range(33):
    #     for fr in range(8):
    #             label[ann, fr] = 1  # binary classification
    # label = torch.from_numpy(label).cuda()
    # print("label shape is {}".format(label.shape))
    #
    # logits = logits.squeeze()
    # clc_loss = F.binary_cross_entropy_with_logits(logits, label)
    # print("clc loss is {}".format(clc_loss))
