'''

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

}
'''

import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import GCNConv, GraphNorm

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='gig_newbone')
class GIGNewBoneDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.backbone = self.build_backbone(config)
        detector_config = config['detector_config']
        self.cls = self.classifier(detector_config)
        self.loss_func = self.build_loss(config)
        self.config = config

    def build_backbone(self, config):
        backbone_name = config['backbone_name']

        if backbone_name == 'swin':
            import timm, os
            model_name = 'swin_base_patch4_window7_224_in22k'
            model = timm.create_model(model_name, pretrained=False)
            ckpt_folder = '/root/wxy/DeepfakeDetection/DeepfakeBench-main/training/pretrained/swin_base_patch4_window7_224.ms_in22k'
            ckpt_path = os.path.join(ckpt_folder, 'pytorch_model.bin')
            sd = torch.load(ckpt_path, map_location='cpu')
            new_sd = {}
            for k, v in sd.items():
                if k.startswith('model.'):
                    new_sd[k[6:]] = v
                else:
                    new_sd[k] = v
            model.load_state_dict(new_sd, strict=False)
            # model = apply_svd_residual_to_swin(model, r_offset=1)
            class _SwinFeatOnly(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x):
                    return self.m.forward_features(x)
            backbone = _SwinFeatOnly(model)
        
        
        return backbone


    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        bs, t, c, h, w = data_dict['image'].shape
        
        if self.config['backbone_name'] == 'swin':
            frame_input = data_dict['image'].reshape(-1, c, h, w)  # (B*T, 3, 224, 224)
            outputs = self.backbone(frame_input)
            outputs = outputs.permute(0, 3, 1, 2)
            features = outputs
        
        
        
        bs_t, c, h, w = features.shape
        features = features.view(bs, t, c, h, w)
        pred, batch_out = self.cls(features)
        return pred, batch_out

    def classifier(self, config): 
        model = GIG_Model(num_class=config['num_class'], num_segment=config['clip_size'], setup_config=config['setup'])
        return model

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label'].long()
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        # we dont compute the video-level metrics for training
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        pred, batch_out = self.features(data_dict)
        prob = torch.softmax(pred, dim=1)[:, 1]
        pred_dict = {'cls': pred, 'prob': prob, 'feat': batch_out}

        return pred_dict


class GIG_Model(nn.Module):
    def __init__(self,
                 num_class=2,
                 num_segment=8,
                 setup_config = None,
                 **kwargs):
        super().__init__()
        self.num_class = num_class
        self.num_segment = num_segment
        self.build_model(setup_config)

    def build_model(self, config):

        self.base_model = GIGFake(pixel_in_channels = config['pixel_in_channels'],
                               pixel_hidden_channels = config['pixel_hidden_channels'],
                               pixel_out_channels = config['pixel_out_channels'],
                               num_classes = self.num_class,
                               edge_refine = config['edge_refine'],
                               feature_dim = config['feature_dim'],)
                               
    def forward(self, x):
    
        img_channel = 3

        out, batch_out = self.base_model(x)

        out = out.view(-1, self.num_segment, self.num_class)  # [n, t, num_class]
        out = out.mean(1, keepdim=False)  # [n, num_class]

        return out, batch_out



class GeneralGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GeneralGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hidden_channels))
        self.layers.append(GCNConv(hidden_channels, hidden_channels))
        self.layers.append(GCNConv(hidden_channels, out_channels))


    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            x = F.relu(x)
        x = self.layers[-1](x, edge_index)
        return x


class EdgeIndexGenerator(nn.Module):
    def __init__(self, in_channels):
        super(EdgeIndexGenerator, self).__init__()
        self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, node_features, height=None, width=None):
        num_nodes = node_features.size(0)
        edge_index = []

        if height and width:
            for i in range(num_nodes):
                batch_idx = i // (height * width)
                pixel_idx = i % (height * width)
                x = pixel_idx % width
                y = pixel_idx // width
                edge_index.append([i, i])
                if x > 0:
                    left_idx = batch_idx * height * width + y * width + (x - 1)
                    edge_index.append([i, left_idx])
                    edge_index.append([left_idx, i])
                if x < width - 1:
                    right_idx = batch_idx * height * width + y * width + (x + 1)
                    edge_index.append([i, right_idx])
                    edge_index.append([right_idx, i])
                if y > 0:
                    up_idx = batch_idx * height * width + (y - 1) * width + x
                    edge_index.append([i, up_idx])
                    edge_index.append([up_idx, i])
                if y < height - 1:
                    down_idx = batch_idx * height * width + (y + 1) * width + x
                    edge_index.append([i, down_idx])
                    edge_index.append([down_idx, i])

            center_x = (width - 1) / 2
            center_y = (height - 1) / 2
            symmetric_x = int(2 * center_x - x)
            symmetric_y = int(2 * center_y - y)
            symmetric_idx = batch_idx * height * width + symmetric_y * width + symmetric_x
            if symmetric_idx != i:
                edge_index.append([i, symmetric_idx])
                edge_index.append([symmetric_idx, i])

        else:
            for i in range(num_nodes):
                edge_index.append([i, i])
                if i > 0:
                    edge_index.append([i, i - 1])
                    edge_index.append([i - 1, i])
                if i < num_nodes - 1:
                    edge_index.append([i, i + 1])
                    edge_index.append([i + 1, i])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index.to(node_features.device)

class GIGFake(nn.Module):
    def __init__(self, pixel_in_channels, pixel_hidden_channels, pixel_out_channels,
                 num_classes, edge_refine, feature_dim):
        super(GIGFake, self).__init__()
        
        self.first_convs = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 2, feature_dim // 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 4 , pixel_in_channels[0], 3, 1, 1),
        )

        num_iterations = len(pixel_in_channels)

        batch_in_channels = [_ * 7 * 7 for _ in pixel_out_channels]
        batch_hidden_channels = [_ // 49 for _ in batch_in_channels]
        batch_out_channels = [_ * 7 * 7 for _ in pixel_out_channels]

        self.pixel_gcns = nn.ModuleList([
            GeneralGCN(pixel_in_channels[i], pixel_hidden_channels[i], pixel_out_channels[i])
            for i in range(num_iterations)
        ])

        self.batch_gcns = nn.ModuleList([
            GeneralGCN(batch_in_channels[i], batch_hidden_channels[i], batch_out_channels[i])
            for i in range(num_iterations)
        ])

        self.pixel_edge_generator = EdgeIndexGenerator(pixel_in_channels[0])
        self.batch_edge_generator = EdgeIndexGenerator(batch_in_channels[0])


        self.first_run = True
        self.pixel_edge_index = None
        self.batch_edge_index = None
        self.edge_refine = edge_refine
        self.classifier = nn.Sequential(
            nn.Linear(batch_out_channels[-1], num_classes),
        )

        self.num_iterations = num_iterations

    def forward(self, features):
        a = features
        self.first_run = True
        if len(features.shape) == 5:
            main_batch_size, num, feature_channels, height, width = features.shape
            features = features.view(main_batch_size * num, feature_channels, height, width)
        else:
            main_batch_size = 1
            num, feature_channels, height, width = features.shape

        features = self.first_convs(features)

        _, channels, _, _ = features.shape

        pixel_features = features.permute(0, 2, 3, 1).reshape(-1,
                                                              channels)  # b*n, c, w, h -> b*n, w, h, c -> b*n*w*h, c

        for i in range(self.num_iterations):

            pixel_gcn = self.pixel_gcns[i]
            batch_gcn = self.batch_gcns[i]

            if self.first_run:
                self.pixel_edge_index = self.pixel_edge_generator(pixel_features, height, width)
            elif self.edge_refine:
                print("edge_refine is not available, please set edge_refine = False")

            pixel_output = pixel_gcn(pixel_features, self.pixel_edge_index)  # b*n*w*h, c
            batch_features = pixel_output.view(main_batch_size * num, -1)  # b*n, w*h*c

            if self.first_run:
                self.batch_edge_index = self.batch_edge_generator(batch_features)
                self.first_run = False
            elif self.edge_refine:
                print("edge_refine is not available, please set edge_refine = False")

            batch_output = batch_gcn(batch_features, self.batch_edge_index)  # b*n, w*h*c

            if i < self.num_iterations - 1:
                # batch -> pixel
                pixel_features = batch_output.view(main_batch_size * num * height * width, -1)  # b*n*w*h, c

        logits = self.classifier(batch_output)  # (bs * num, num_classes)
        return logits, batch_output