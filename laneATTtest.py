from ctypes import sizeof
from doctest import testmod
import math
from tkinter import Variable
from turtle import back, forward

import cv2
from sympy import im
import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18, resnet34
from resnet import resnet122 as resnet122_cifar

class TestLaneATT(nn.Module):
    def __init__(self,
                 S=72,  # Npts
                 img_w=640,
                 img_h=360,
                 anchor_feat_channels=64) -> None:
        super(TestLaneATT, self).__init__()
        
        self.feature_extractor = resnet122_cifar()
        self.stride = 4
        backbone_nb_channels = 64
        self.img_w = img_w
        self.n_strips = S - 1   # 71
        self.n_offsets = S  # 72
        self.fmap_h = img_h // self.stride  # 90
        fmap_w = img_w // self.stride   # 160
        self.fmap_w = fmap_w
        self.anchor_ys = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)    # [72]
        self.anchor_cut_ys = torch.linspace(1, 0, steps=self.fmap_h, dtype=torch.float32)   # [90]  
        self.anchor_feat_channels = anchor_feat_channels

        self.left_angles = [72., 60., 49., 39., 30., 22.]   # 6
        self.right_angles = [108., 120., 131., 141., 150., 158.]    # 6
        self.bottom_angles = [165., 150., 141., 131., 120., 108., 100., 90., 80., 72., 60., 49., 39., 30., 15.]     # 15

        self.anchors, self.anchors_cut = self.generate_anchors(lateral_n=72, bottom_n=128)

        # Pre compute indices for the anchor pooling
        self.cut_zs, self.cut_ys, self.cut_xs, self.invalid_mask = self.compute_anchor_cut_indices(
            self.anchor_feat_channels, fmap_w, self.fmap_h)

        self.conv1 = nn.Conv2d(backbone_nb_channels, self.anchor_feat_channels, kernel_size=1)
        self.cls_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, 2)
        self.reg_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, self.n_offsets + 1)
        self.attention_layer = nn.Linear(self.anchor_feat_channels * self.fmap_h, len(self.anchors) - 1)
        self.initialize_layer(self.attention_layer)
        self.initialize_layer(self.conv1)
        self.initialize_layer(self.cls_layer)
        self.initialize_layer(self.reg_layer)

    def generate_anchors(self, lateral_n, bottom_n):
        left_anchors, left_cut = self.generate_side_anchors(self.left_angles, x=0., nb_origins=lateral_n)   # [432, 77], [432, 95]
        right_anchors, right_cut = self.generate_side_anchors(self.right_angles, x=1., nb_origins=lateral_n)    # [432, 77], [432, 95]
        bottom_anchors, bottom_cut = self.generate_side_anchors(self.bottom_angles, y=1., nb_origins=bottom_n)  # [1920, 77], [1920, 95]
        return torch.cat([left_anchors, bottom_anchors, right_anchors]), torch.cat([left_cut, bottom_cut, right_cut])   # [2784, 77], [2784, 95]

    def generate_side_anchors(self, angles, nb_origins, x=None, y=None):
        if x is None and y is not None:
            starts = [(x, y) for x in np.linspace(1., 0., num=nb_origins)]  # bottom line anchors' starts
        elif x is not None and y is None:
            starts = [(x, y) for y in np.linspace(1., 0., num=nb_origins)]  # lateral line anchors' starts
        else:
            raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')
        n_anchors = nb_origins * len(angles)    # left, right: 72 * 6 = 432; bottom: 128 * 15 = 1920
        # each row, first for x and second for y:
        # 2 scores, 1 start_y, start_x, 1 lenght, S coordinates, score[0] = negative prob, score[1] = positive prob
        anchors = torch.zeros((n_anchors, 2 + 2 + 1 + self.n_offsets))
        anchors_cut = torch.zeros((n_anchors, 2 + 2 + 1 + self.fmap_h))
        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k = i * len(angles) + j     # k = (0, ..., n_anchors-1)
                anchors[k] = self.generate_anchor(start, angle)
                anchors_cut[k] = self.generate_anchor(start, angle, cut=True)

        return anchors, anchors_cut

    def generate_anchor(self, start, angle, cut=False):
        if cut:
            anchor_ys = self.anchor_cut_ys  # [90]
            anchor = torch.zeros(2 + 2 + 1 + self.fmap_h)   # [95]
        else:
            anchor_ys = self.anchor_ys  # [72]  
            anchor = torch.zeros(2 + 2 + 1 + self.n_offsets) # [77]
        angle = angle * math.pi / 180.  # degrees to radians
        start_x, start_y = start
        anchor[2] = 1 - start_y     # anchor[0], anchor[1]:scores[0], scores[1]; anchor[4]:length
        anchor[3] = start_x
        anchor[5:] = (start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)) * self.img_w
        return anchor

    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        batch_features = self.feature_extractor(x)
        batch_features = self.conv1(batch_features)
        print("feature map shape = {}".format(batch_features.shape))
        batch_anchor_features = self.cut_anchor_features(batch_features)
        print("generate anchor features shape = {}".format(batch_anchor_features.shape))

        # Join proposals from all images into a single proposals features batch
        batch_anchor_features = batch_anchor_features.view(-1, self.anchor_feat_channels * self.fmap_h)
        print(batch_anchor_features.shape)

        # Add attention features
        softmax = nn.Softmax(dim=1)
        scores = self.attention_layer(batch_anchor_features)
        attention = softmax(scores).reshape(x.shape[0], len(self.anchors), -1)
        attention_matrix = torch.eye(attention.shape[1], device=x.device).repeat(x.shape[0], 1, 1)
        non_diag_inds = torch.nonzero(attention_matrix == 0., as_tuple=False)
        attention_matrix[:] = 0
        attention_matrix[non_diag_inds[:, 0], non_diag_inds[:, 1], non_diag_inds[:, 2]] = attention.flatten()
        batch_anchor_features = batch_anchor_features.reshape(x.shape[0], len(self.anchors), -1)
        attention_features = torch.bmm(torch.transpose(batch_anchor_features, 1, 2),
                                       torch.transpose(attention_matrix, 1, 2)).transpose(1, 2)
        attention_features = attention_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        batch_anchor_features = batch_anchor_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        batch_anchor_features = torch.cat((attention_features, batch_anchor_features), dim=1)
        print("after attention features shape = {}".format(batch_anchor_features.shape))

        # Predict
        cls_logits = self.cls_layer(batch_anchor_features)
        reg = self.reg_layer(batch_anchor_features)
        print("cls.shape = {}".format(cls_logits.shape))
        print("reg.shape = {}".format(reg.shape))

        # Undo joining
        cls_logits = cls_logits.reshape(x.shape[0], -1, cls_logits.shape[1])
        reg = reg.reshape(x.shape[0], -1, reg.shape[1])

        # Add offsets to anchors
        reg_proposals = torch.zeros((*cls_logits.shape[:2], 5 + self.n_offsets), device=x.device)
        reg_proposals += self.anchors
        reg_proposals[:, :, :2] = cls_logits
        reg_proposals[:, :, 4:] += reg

        print(reg_proposals.shape)
    
    def cut_anchor_features(self, features):
        # definitions
        batch_size = features.shape[0]
        n_proposals = len(self.anchors)
        n_fmaps = features.shape[1]
        batch_anchor_features = torch.zeros((batch_size, n_proposals, n_fmaps, self.fmap_h, 1), device=features.device)
        # actual cutting
        for batch_idx, img_features in enumerate(features):
            rois = img_features[self.cut_zs, self.cut_ys, self.cut_xs].view(n_proposals, n_fmaps, self.fmap_h, 1)   # 2784, 64, 90, 1
            rois[self.invalid_mask] = 0
            batch_anchor_features[batch_idx] = rois

        return batch_anchor_features

    def compute_anchor_cut_indices(self, n_fmaps, fmaps_w, fmaps_h):
        # definitions
        n_proposals = len(self.anchors_cut)

        # indexing
        unclamped_xs = torch.flip((self.anchors_cut[:, 5:] / self.stride).round().long(), dims=(1,))
        unclamped_xs = unclamped_xs.unsqueeze(2)
        unclamped_xs = torch.repeat_interleave(unclamped_xs, n_fmaps, dim=0).reshape(-1, 1)
        cut_xs = torch.clamp(unclamped_xs, 0, fmaps_w - 1)
        unclamped_xs = unclamped_xs.reshape(n_proposals, n_fmaps, fmaps_h, 1)
        invalid_mask = (unclamped_xs < 0) | (unclamped_xs > fmaps_w)
        cut_ys = torch.arange(0, fmaps_h)
        cut_ys = cut_ys.repeat(n_fmaps * n_proposals)[:, None].reshape(n_proposals, n_fmaps, fmaps_h)
        cut_ys = cut_ys.reshape(-1, 1)
        cut_zs = torch.arange(n_fmaps).repeat_interleave(fmaps_h).repeat(n_proposals)[:, None]

        return cut_zs, cut_ys, cut_xs, invalid_mask

if __name__ == "__main__":
    print("test begin")
    testModel = TestLaneATT()
    img = np.random.random((1,3,360,640))
    input = torch.tensor(data=img, dtype=torch.float32)
    testModel.forward(x=input)