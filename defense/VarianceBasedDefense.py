#!/usr/bin/env python3

# Software Name: BackdoorBench/VarianceBasedDefense
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This software is distributed under the Creative Commons Attribution Non Commercial 4.0 International,
# see the "LICENSE.txt" file for more details
#
# Authors: Sujeevan Aseervatham
# Software description: Implementation of the variance-based defense method

import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
from typing import Union
from torch.utils.data.dataloader import DataLoader

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from defense.base import defense

from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import PureCleanModelTrainer, BackdoorModelTrainer
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform, get_dataset_normalization
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.nCHW_nHWC import *

from utils.bd_dataset_v2 import get_labels, prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform

from sklearn.cluster import KMeans

from scipy.stats import ks_2samp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import transforms
import copy

import PIL, cv2, json

class SimpleClassifierBlock(nn.Module):
    def __init__(self, in_channels, n_filters, use_bn, dropout_perc, use_pooling=False):
        super().__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(in_channels, n_filters, (3,3), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.pool = nn.MaxPool2d(2, 2) if use_pooling else None
        self.dropout = nn.Dropout(dropout_perc) if dropout_perc>0 else None
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = nn.functional.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class SimpleClassifierModel(nn.Module):
    def __init__(self, img_height, img_width, classes, arch_str=None, n_blocks=1, n_filters=32, use_bn=False, dropout_perc=0., output_prob=False, use_sigmoid=False):
        #img_height and img_width should be even numbers
        super().__init__()
        self.blocks = nn.Sequential()
        if arch_str is not None:
            filters_str = arch_str.split('-')
            prev_filter_size = 3
            n_blocks = len(filters_str)
            h_out, w_out = img_height, img_width
            for i, f in enumerate(filters_str):
                dropout = 0 if i != (n_blocks -1) else dropout_perc #dropout only for the last layer
                pooling = False
                if f[-1] == 'P':
                    pooling = True
                    f = f[:-1]
                num_filters = int(f)
                self.blocks.add_module(f"conv{i+1}", SimpleClassifierBlock(prev_filter_size, num_filters, use_bn, dropout, pooling))
                prev_filter_size = num_filters
                pad, dilat, kernel_size, stride = 0, 1, 3, 1
                h_out = int(np.floor((h_out + 2*pad - dilat * (kernel_size -1) -1) / stride ) + 1)
                w_out = int(np.floor((w_out + 2*pad - dilat * (kernel_size -1) -1) / stride ) + 1)
                if pooling == True:
                    pad, dilat, kernel_size, stride = 0, 1, 2, 2
                    h_out = int(np.floor((h_out + 2*pad - dilat * (kernel_size -1) -1) / stride ) + 1)
                    w_out = int(np.floor((w_out + 2*pad - dilat * (kernel_size -1) -1) / stride ) + 1)
            n_filters = prev_filter_size
        else:
            h_out, w_out = img_height, img_width
            for i in range(0, n_blocks):
                nfilts = 3 if i==0 else n_filters
                if (i+1) == n_blocks:
                    dropout, pooling = dropout_perc, True  #dropout and pooling only for the last layer
                else:
                    dropout, pooling = 0., False
                self.blocks.add_module(f"conv{i+1}", SimpleClassifierBlock(nfilts, n_filters, use_bn, dropout, pooling))
                pad, dilat, kernel_size, stride = 0, 1, 3, 1
                h_out = int(np.floor((h_out + 2*pad - dilat * (kernel_size -1) -1) / stride ) + 1)
                w_out = int(np.floor((w_out + 2*pad - dilat * (kernel_size -1) -1) / stride ) + 1)
                if pooling == True:
                    pad, dilat, kernel_size, stride = 0, 1, 2, 2
                    h_out = int(np.floor((h_out + 2*pad - dilat * (kernel_size -1) -1) / stride ) + 1)
                    w_out = int(np.floor((w_out + 2*pad - dilat * (kernel_size -1) -1) / stride ) + 1)
        self.fc1 = nn.Linear(n_filters * h_out * w_out, classes)
        self.use_bn = use_bn
        self.prob_func = None
        if output_prob:
            self.prob_func = nn.Sigmoid() if (use_sigmoid or(classes == 1)) else nn.Softmax(dim=1)
    def to(self, device, dtype=None, non_blocking=False):
        t = super().to(device=device, dtype=dtype, non_blocking=non_blocking)
        t.blocks = self.blocks.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return t
    def forward(self, x):
        x = self.blocks(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        if self.prob_func is not None:
            x = self.prob_func(x) 
        return x
    
class EnsembleModel:
    def __init__(self, img_height, img_width, classes, models:Union[list,str] = []):
        self.img_height = img_height
        self.img_width = img_width
        self.classes = classes
        if type(models) is str:
            self.models = []
            self.add(models)
        else:
            self.models = models
    def add(self, model:Union[nn.Module,str]):
        if type(model) is str:
            for arch in model.split(','):
                self.models.append( SimpleClassifierModel(self.img_height, self.img_width, self.classes, use_bn=True, arch_str=arch) ) #use bn by default
        else:
            self.models.append(model)
    def to(self, device, dtype=None, non_blocking=False):
        for i, m in enumerate(self.models):
            self.models[i] = m.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self
    def train(self):
        for m in self.models:
            m.train()
        return self
    def eval(self):
        for m in self.models:
            m.eval()
        return self
    def __len__(self):
        return len(self.models)
    def __getitem__(self, i):
        return self.models[i]
    def __iter__(self):
        return self.models.__iter__()
    def __call__(self, x, transf=None, prob_func=None, no_grad=False):
        if transf is None:
            return [self.models[i](x) for i in range(len(self.models))]
        elif not isinstance(transf, (list, tuple)):
            x = transf(x)
            return [self.models[i](x) for i in range(len(self.models))]
        else:
            preds = []
            for i in range(len(self.models)):
                xt = x
                if isinstance(transf[i], (list, tuple)):
                    for t in transf[i]:
                        xt = t(xt)
                else:
                    xt = transf[i](xt)
                if no_grad:
                    with torch.no_grad():
                        pred = self.models[i](xt)
                        if prob_func is not None:
                            pred = prob_func(pred)
                        preds.append(pred)
                else:
                    pred = self.models[i](xt)
                    if prob_func is not None:
                        pred = prob_func(pred)
                    preds.append(pred)
            return preds
    
    def read_from_file(self, filename):
        d = torch.load(filename, weights_only=True)
        for i, m  in enumerate(self.models):
            m.load_state_dict(d[i])
            m.eval()
    def write_to_file(self, filename):
        torch.save([m.state_dict() for m in self.models], filename)
    
class GradTransf(nn.Module):
    def __init__(self, pois_model, poisoned_class):
        super().__init__()
        self.pois_model = pois_model
        self.poisoned_class = poisoned_class
    def forward(self, x):
        return self.get_grad_multiple(self.pois_model, x, self.poisoned_class)
    def to(self, device, dtype=None, non_blocking=False):
        t = super().to(device=device, dtype=dtype, non_blocking=non_blocking)
        t.pois_model = t.pois_model.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return t
    @staticmethod
    def get_grad_multiple(model, x, class_label):
        device = next(model.parameters()).device.type
        im = x.detach().clone()
        im.requires_grad = True
        model.eval()
        pred = model(im)
        #ll = 2*pred[:, class_label] - torch.sum(pred, dim=1)  #loss function
        n_inst, n_classes= pred.shape[0], pred.shape[1]
        t = torch.arange(n_classes).repeat(n_inst,1)!=class_label
        ll = pred[:,class_label] - torch.max(pred[t].view((n_inst, n_classes-1)), dim=1)[0]

        ll.backward(torch.Tensor([1]*len(im)).to(device))
        g = im.grad.data.detach() #.squeeze()
        #grad_max = torch.amax(torch.abs(g), dim=(1,2,3))
        #g = torch.sqrt(torch.sum(g**2, dim=1))
        #g_norm = (g-g.amin(dim=(1,2)).reshape(g.shape[0],1,1))/(g.amax(dim=(1,2)) - g.amin(dim=(1,2))+1e-6).reshape(g.shape[0],1,1)
        return g

class PatternAdder(nn.Module):
    def __init__(self, mask, pattern, alpha=1.0):
        super().__init__()
        self.mask = mask
        self.pattern = pattern
        self.alpha = alpha
        self.set_random_gen()
    def set_random_gen(self, rnd_gen=None):
        if rnd_gen is None:
            rnd_gen = np.random
        self.rnd_gen = rnd_gen
    def to(self, device, dtype=None, non_blocking=False):
        t = super().to(device=device, dtype=dtype, non_blocking=non_blocking)
        t.mask = t.mask.to(device=device, dtype=dtype, non_blocking=non_blocking)
        t.pattern = t.pattern.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return t
    def inject_pattern_with_random_alpha(self, x):
        x_shape= x.shape
        n = x_shape[0]
        alpha_min, alpha_max = self.alpha
        alpha = torch.tensor([alpha_min]*n)
        if (alpha_min<alpha_max):
            alpha = torch.tensor(self.rnd_gen.uniform(alpha_min, alpha_max, len(x)), dtype=torch.float32)
        alpha = alpha.unsqueeze(1).to(x.device.type)
        x = (x*(1-self.mask)).view(n, -1)  + ( (1.-alpha)*(x*(self.mask)).view(n, -1)) + (alpha *(self.pattern*self.mask).unsqueeze(3).tile(n).permute(3,0,1,2).view(n,-1))
        x = x.view(x_shape)
        return x

    def forward(self, x):
        if type(self.alpha) in [tuple, list, np.ndarray]:
            x = self.inject_pattern_with_random_alpha(x)
        else:
            x = (x * (1-self.mask)) + (1.-self.alpha)*(x * self.mask) + self.alpha*(self.pattern*self.mask)
        return x


class relabeled_dataset_wrapper_with_transform(dataset_wrapper_with_transform):
    def __init__(self, obj, wrap_img_transform=None, wrap_label_transform=None, index_label_dict={}):
        super().__init__(obj, wrap_img_transform, wrap_label_transform)
        self.index_label_dict = index_label_dict
    def __getitem__(self, index):
        img, label, *other_info = self.wrapped_dataset[index]
        if self.wrap_img_transform is not None:
            img = self.wrap_img_transform(img)
        if self.wrap_label_transform is not None:
            label = self.wrap_label_transform(label)
        label = self.index_label_dict.get(index, label)
        return (img, label, *other_info)

class VarianceBasedDefense(defense):

    def __init__(self,args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"
        
        self.args = args
        self.eps = 1e-6

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', default = False, type=lambda x: str(x) in ['True','true','1'])

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='the location of result')
    
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--model', type=str, help='resnet18')
        
        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')

        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/defense/vbd/config.yaml", help='the path of yaml')

        #set the parameter for the defense
        parser.add_argument('--pois_model_arch', type=str, help='Poisoned model arch, i.e. 64P,10-10P,16-16P')
        parser.add_argument('--pois_train_perc', type=float, help='Perc of the training data to use for the poisoned model')
        parser.add_argument('--ensemble_detect_min_thr', type=int, help='Min number of models that must consider the class as poisoned (majority vote for robustness or 1 to reduce the FN)')
        parser.add_argument('--pois_max_epochs', type=int, help='Max epochs for the training of the poisoned model')
        parser.add_argument('--max_instances_for_variance', type=int, help='Max number of instances to compute the variance')
        parser.add_argument('--min_instances_for_variance', type=int, help='min number of instances to compute the variance')
        parser.add_argument('--pois_class_detect_alpha_thr',type=float, help='Threshold used fo the KS stat test to detect the poisoned classes')
        parser.add_argument('--pattern_min_asr', type=float, help='min asr value for a pattern to be detected as an attack pattern')

        parser.add_argument('--detector_train_perc', type=float, help='perc. of the training data to use for the training of the pattern detector model')
        parser.add_argument('--detector_max_epochs', type=int, help='Max number of epochs for the training of the pattern detector model')
        parser.add_argument('--retrain_pois_method',  type=str, help='Method to deal with the detected poison for the retrain: suppress, relabel_with_det_pois_model, relabel_with_pois_model')
        parser.add_argument('--no_retrain',  type=str, help='True if use only detection')


    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/defense/vbd/'
        if not (os.path.exists(save_path)):
                os.makedirs(save_path) 
        # assert(os.path.exists(save_path))    
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save) 
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)  
        self.result = load_attack_result(attack_file + '/attack_result.pt')

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model = model,
        )

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')
    
    def set_devices(self):
        # self.device = torch.device(
        #     (
        #         f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
        #         # since DataParallel only allow .to("cuda")
        #     ) if torch.cuda.is_available() else "cpu"
        # )
        self.device = self.args.device

    @staticmethod
    def strat_subset(labels, sub_perc):
        subs = []
        for l in np.unique(labels):
            linds = np.where(labels == l)[0]
            t = np.random.choice(linds, size=int(round(len(linds)*sub_perc)), replace=False, p=None)
            subs.append(t)
        return np.concatenate(subs)
    
    def log_info(self, message, toprt=True):
        if toprt:
            print(message)
        logging.info(message)

    def move_to_device(self, m):
        if "," in self.device:
            m = torch.nn.DataParallel(
                m,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{m.device_ids[0]}'
            m = m.to(self.args.device)
        else:
            m = m.to(self.args.device)
        return m
    
    def stage_1_split_train_detect(self, train_labels, pois_train_perc, pois_model_path):
        train_indexes_path = pois_model_path+ f'pois_train_indexes_{pois_train_perc}.txt'
        if os.path.exists(train_indexes_path):
            _m = f'File {train_indexes_path} exists: reading indexes'
            logging.info(_m); #print(_m)
            pois_train_indexes = np.loadtxt(train_indexes_path, dtype=int)
        else:
            pois_train_indexes = self.strat_subset(train_labels, pois_train_perc)
            np.savetxt(train_indexes_path, pois_train_indexes, delimiter='\n', fmt='%i')
        pois_detect_indexes = np.setdiff1d(np.arange(len(train_labels)), pois_train_indexes)
        return pois_train_indexes, pois_detect_indexes
    

    def stage_1_learn_poison_model(self, train_dataset, train_labels, pois_train_indexes, pois_validation_indexes, test_dataset, bd_test_dataset, pois_model_path):
        pois_model_weight = pois_model_path + 'pois_model.pt'
        pois_model_acc_fname = pois_model_path + 'pois_model_acc.csv'

        original_index_train_dataset = train_dataset.wrapped_dataset.original_index_array
        orig_train_trans = train_dataset.wrap_img_transform
        orig_test_trans = test_dataset.wrap_img_transform
        orig_bd_test_trans = bd_test_dataset.wrap_img_transform

        n_poison_inst = np.sum([train_dataset.poison_indicator[i] for i in train_dataset.wrapped_dataset.original_index_array[pois_train_indexes]])
        _m = f"Stage 1: Learning the poison model on {self.args.pois_train_perc*100}% of the data...\n" \
            +f"Nb of training data: {len(pois_train_indexes)} ({int(100*len(pois_train_indexes)/len(train_dataset))}%)\n" \
            +f"Nb of poison instances: {n_poison_inst} ({int(100*n_poison_inst/len(pois_train_indexes))}%)\n" \
            +f"Nb of clean test data: {len(self.result['clean_test'])}\n" \
            +f"Nb of bad test data: {len(self.result['bd_test'])}"
        logging.info(_m); #print(_m)

        pois_model = EnsembleModel(self.args.input_height, self.args.input_width, self.args.num_classes, models=[])
        pois_model.add(self.args.pois_model_arch)
        pois_model = self.move_to_device(pois_model)
        pois_model.train()

        optimizer_list = []
        for i, m in enumerate(pois_model):
            optimizer, scheduler = argparser_opt_scheduler(pois_model[i], self.args)
            optimizer_list.append( (optimizer, scheduler) )

        normal_trans = transforms.Compose([transforms.ToTensor(), get_dataset_normalization(self.args.dataset)])
        train_dataset.wrap_img_transform = normal_trans
        test_dataset.wrap_img_transform = normal_trans
        bd_test_dataset.wrap_img_transform = normal_trans

        train_poison_flags = np.array(train_dataset.poison_indicator)[pois_train_indexes].astype(bool)
        valid_poison_flags = np.array(train_dataset.poison_indicator)[pois_validation_indexes].astype(bool)
        valid_labels = train_labels[pois_validation_indexes]

        validation_dataset = dataset_wrapper_with_transform(train_dataset.wrapped_dataset.copy(), normal_trans, train_dataset.wrap_label_transform)
        validation_dataset.subset(pois_validation_indexes)
        train_dataset.subset(pois_train_indexes)
        train_labels = train_labels[pois_train_indexes]

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True,
                                  pin_memory=self.args.pin_memory, num_workers=self.args.num_workers, )
        train_loader_eval = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False,
                                  pin_memory=self.args.pin_memory, num_workers=self.args.num_workers, )
        valid_loader = DataLoader(validation_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False,
                                  pin_memory=self.args.pin_memory, num_workers=self.args.num_workers, )

        pois_model_acc = []
        if os.path.exists(pois_model_weight) and os.path.exists(pois_model_acc_fname):
            logging.info(f'File {pois_model_weight} exists: reading model')
            pois_model.read_from_file(pois_model_weight)
            #read acc file
            with open(pois_model_acc_fname, "r") as f:
                for l in f.readlines()[1:] :  #skip header
                    t = l.split(',')
                    pois_model_acc.append( (float(t[0]), float(t[1])) )
        else:
            criterion = nn.CrossEntropyLoss() if self.args.num_classes>1 else nn.BCEWithLogitsLoss()
            #optimizer, scheduler = argparser_opt_scheduler(pois_model, self.args)
            #optimizer = torch.optim.SGD(pois_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            #optimizer = torch.optim.Adam(pois_model.parameters(), lr=0.01)
            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=2, threshold=0.0001, min_lr=0, eps=1e-08)
            best_loss, best_model_params = None, None 
            for epoch in range(self.args.pois_max_epochs):
                pois_model.train()
                for i, data in enumerate(train_loader, 0):
                    inputs, labels, *oth = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # zero the parameter gradients
                    # forward + backward + optimize
                    for i, m in enumerate(pois_model):
                        o,s = optimizer_list[i]
                        o.zero_grad()
                        l = criterion(m(inputs), labels)
                        l.backward()
                        o.step()

                pois_model.eval()
                logging.info(f"Epoch {epoch}: ")
                train_acc, valid_acc = [], []
                vloss = []
                t = [("train", train_loader_eval, train_labels, train_poison_flags), ("validation", valid_loader, valid_labels, valid_poison_flags)]
                for mtype, tloader, true_labels, true_poison_flags in t:
                    predicted_y = self.predict(pois_model, tloader, prob_func=None, x_transformer=None)
                    for i, pred_y in enumerate(predicted_y):
                        loss = criterion(pred_y, torch.tensor(true_labels).to(self.device)).item()
                        pred_y = torch.argmax(pred_y, dim=1).detach().cpu().numpy()
                        acc = np.sum(pred_y==true_labels)/float(len(pred_y))
                        clean_acc = np.sum(pred_y[~true_poison_flags]==true_labels[~true_poison_flags])/float(np.sum(~true_poison_flags))
                        asr = np.sum(pred_y[true_poison_flags]==true_labels[true_poison_flags])/float(np.sum(true_poison_flags))
                        if mtype == "validation":
                            vloss.append(loss)
                            valid_acc.append(acc)
                        else:
                            train_acc.append(acc)
                        logging.info(f"{mtype} Perf model {i}: acc: {np.round(acc,4)}, clean_acc: {np.round(clean_acc,4)}, asr: {np.round(asr,4)}, loss: {np.round(loss,4)}")
                
                for i, (_, scheduler) in enumerate(optimizer_list):
                    scheduler.step(vloss[i])

            pois_model_acc = [(train_acc[i], valid_acc[i]) for i in range(len(train_acc))]
            
            #pois_model.load_state_dict(best_model_params)
            pois_model.eval()
            pois_model.write_to_file(pois_model_weight)
            #write the acc
            with open(pois_model_acc_fname, "w") as f:
                f.write('train_acc,test_acc\n')
                for t, v in pois_model_acc:
                    f.write(f'{t},{v}\n')

        trainer = BackdoorModelTrainer(pois_model[0])
        trainer.non_blocking = self.args.non_blocking
        trainer.criterion = nn.CrossEntropyLoss() if self.args.num_classes>1 else nn.BCEWithLogitsLoss()
        trainer.model.train()
        t = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False,
                        pin_memory=self.args.pin_memory, num_workers=self.args.num_workers, )
        clean_train_metrics = trainer.test_given_dataloader(t, verbose=1, device=self.device)[0]
        t = DataLoader(self.result['clean_test'], batch_size=self.args.batch_size, shuffle=False, drop_last=False,
                        pin_memory=self.args.pin_memory, num_workers=self.args.num_workers,)
        clean_test_metrics = trainer.test_given_dataloader(t, verbose=1, device=self.device)[0]
        t = DataLoader(self.result['bd_test'], batch_size=self.args.batch_size, shuffle=False, drop_last=False,
                        pin_memory=self.args.pin_memory, num_workers=self.args.num_workers,)
        bd_test_metrics = trainer.test_given_dataloader(t, verbose=1, device=self.device)[0]
        _m = f"Poisoned Model {0} performance:\nTrain acc: {clean_train_metrics['test_acc']}\nTest acc:{clean_test_metrics['test_acc']}\nBad Test acc:{bd_test_metrics['test_acc']}"
        logging.info(_m); #print(_m)

        train_dataset.wrapped_dataset.original_index_array = original_index_train_dataset
        train_dataset.wrap_img_transform = orig_train_trans
        test_dataset.wrap_img_transform = orig_test_trans
        bd_test_dataset.wrap_img_transform = orig_bd_test_trans

        return pois_model, np.array(pois_model_acc)

    
    def detect_poisoned_classes(self, pois_model, detection_set, save_path):
        poisoned_classes_files = save_path+'poisoned_classes.csv'
        if os.path.exists(poisoned_classes_files):
            _m = f'File {poisoned_classes_files} exists: reading the poisoned classes'
            logging.info(_m)

            poisoned_classes, pvalues = [[] for i in range(len(pois_model))], [[] for i in range(len(pois_model))]
            with open(poisoned_classes_files, "r") as f:
                for l in f.readlines()[1:] :  #skip header
                    t = l.split(',')
                    model_ind, pclass, pvalue = int(t[0]), int(t[1]), float(t[2])
                    if pvalue < self.args.pois_class_detect_alpha_thr:
                        poisoned_classes[model_ind].append(pclass)
                        pvalues[model_ind].append(pvalue)
            for i in range( len(pois_model) ):
                logging.info(f'Model {i}: {len(poisoned_classes[i])} poisoned classes read: {poisoned_classes[i]}')
            return poisoned_classes
        
        normal_trans = transforms.Compose([transforms.ToTensor(), get_dataset_normalization(self.args.dataset)])
        original_index = detection_set.wrapped_dataset.original_index_array
        original_trans = detection_set.wrap_img_transform
        detection_set.wrap_img_transform = None
        detection_labels  = np.array(get_labels(detection_set))
        detection_set.wrap_img_transform = normal_trans

        pois_model.eval()
        all_classes, all_pvalues = [[] for i in range(len(pois_model))], [[] for i in range(len(pois_model))]
        poisoned_classes = [[] for i in range(len(pois_model))]
        for cl_label in range(self.args.num_classes):
            detection_set.wrapped_dataset.original_index_array = original_index
            detection_set.subset(np.where(detection_labels==cl_label))
            predicted_y = self.predict(pois_model, detection_set, prob_func=nn.Softmax(dim=1))
            for i, pred_y in enumerate(predicted_y):
                ypred = torch.argmax(pred_y, dim=1).detach().cpu().numpy()
                ind = np.where(ypred==cl_label)[0]
                n = np.min([self.args.max_instances_for_variance, len(ind)]) #we need a max of n instances
                if n<self.args.min_instances_for_variance:
                    logging.info(f'Not enough poisoned data for ks test ({n}<{self.args.min_instances_for_variance})')
                    continue
                ind = ind[torch.sort(pred_y[ind][:,cl_label])[1][-n:].cpu().numpy()] #keep only the instances with the highest values

                #get the instance of the class with the lowest probability
                ind_low = np.arange(len(ypred))
                ind_low = np.setdiff1d(ind_low, ind, assume_unique=True) #remove the top instances
                n_low = np.min([self.args.max_instances_for_variance, len(ind_low)])
                ind_low = ind_low[torch.sort(pred_y[ind_low][:,cl_label])[1][:n_low].cpu().numpy()]

                img_set = torch.stack([detection_set[i][0] for i in ind])
                img_set_low = torch.stack([detection_set[i][0] for i in ind_low])
                gradtrans = GradTransf(pois_model[i], cl_label)
                g = gradtrans(img_set.to(self.device))
                m = torch.max(torch.abs(g.mean(dim=0)))
                if torch.any(torch.isnan(m)):
                    logging.info(f'Model {i}: NaN grads (Rejecting for ks test)')
                    continue
                gflags = (torch.abs(g.mean(dim=0)) == m)  #get the coords of the max values
                pois_stat = torch.abs(g[:, gflags]).amax(dim=1) #get the max values for the top sample

                g_low = gradtrans(img_set_low.to(self.device))
                non_pois_stat = torch.abs(g_low[:, gflags]).amax(dim=1) #get the max values for the bottom sample

                #compute the 2sided ks test for the two samples
                r = ks_2samp(pois_stat.detach().cpu().numpy(), non_pois_stat.detach().cpu().numpy(), alternative='two-sided')
                all_classes[i].append(cl_label)
                all_pvalues[i].append(r.pvalue)
                if r.pvalue < self.args.pois_class_detect_alpha_thr:
                    poisoned_classes[i].append(cl_label)

        for i in range(len(poisoned_classes)):
            logging.info(f'Model {i}: {len(poisoned_classes[i])} poisoned classes detected: {poisoned_classes[i]}')

        detection_set.wrapped_dataset.original_index_array = original_index
        detection_set.wrap_img_transform = original_trans

        with open(poisoned_classes_files, 'w') as f:
            f.write('model,class,pvalue\n')
            for im in range(len(pois_model)):
                for i in range(len(all_classes[im])):
                    f.write(f'{im},{all_classes[im][i]},{all_pvalues[im][i]}\n')
        
        return poisoned_classes

    
    def get_mad_threshold_value(self, data, mad_factor=2):
        #get the median absolute deviation upper bound
        if type(data) == list:
            data= np.array(data)
        if type(data) == np.ndarray:
            med = np.median(data)
            mad = 1.4826*np.median(np.abs(data-med)) #median of the absolute deviation with a scale factor 1.4826
        else:
            med = torch.median(data).item()
            mad = 1.4826*torch.median(torch.abs(data - med)).item()
        return med + mad_factor*mad

    def otsu_binarize(self, img):
        if type(img) == torch.tensor:
            img = img.detach().cpu()
        v = (img - img.min())/(img.max()-img.min()+self.eps)
        v = (v*255).cpu().numpy()
        v = np.round(v).astype(np.uint8)
        _, otsu_thresholding = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return torch.tensor(otsu_thresholding/255.0, dtype=torch.float32)

    def gaussian_blur(self, img):
        c = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), stride=1, padding=0, bias=False)
        pad = nn.ReflectionPad2d(1).to(img.device)
        c.weight.data = (torch.tensor([[1,2,1],[2,4,2], [1,2,1]])*1/16.0).unsqueeze(0).unsqueeze(0)
        c = c.to(img.device)
        m = c(pad(img.unsqueeze(0))).squeeze(dim=0).detach().cpu()
        return m
    
    def compute_var(self, img_set, perc_pixel_thresh=0.3):
        #img_set shape (batch, h, w, c)
        t = torch.var(img_set,0) #compute the var for each pixel
        m = torch.tensor([t[:,:,i].min() for i in range(t.shape[2])])
        max = torch.tensor([t[:,:,i].max() for i in range(t.shape[2])])
        t = (t - m) / (max-m +self.eps) #normalize
        t = torch.mean(t, dim=2)  #compute the mean normalize var for each pixel along the rgb axis
        return t

    def get_gradient_mask(self, model, img_set, class_label):
        device = next(model.parameters()).device.type
        im = img_set.detach().clone()
        im.requires_grad = True
        y = torch.tensor([class_label], dtype=torch.int64).to(device)
        model.eval()
        pred = model(im.to(device))
        #ll = torch.mean(2*pred[:,class_label] - torch.sum(pred, dim=0))
        n_inst, n_classes= pred.shape[0], pred.shape[1]
        t = (torch.arange(n_classes).repeat(n_inst,1) != class_label)
        ll = torch.mean( pred[:,class_label] - torch.max(pred[t].view((n_inst, n_classes-1)), dim=1)[0] )
        
        ll.backward()
        g = im.grad.data.detach().squeeze()
        g = g.sum(dim=0).detach() #don't do the mean, it is already done in the loss
        grad_max = torch.max(torch.abs(g))
        #print(f'grad max: {torch.max(torch.abs(g))}')
        g = torch.sqrt(torch.sum(g**2, dim=0))  #compute the norm of RGB grad vec
        g = (g - g.min())/(g.max()-g.min()+self.eps) #normalize the norm to be within [0,1]
        filtered_grad = self.gaussian_blur(g.to(device))
        filtered_grad = self.otsu_binarize(filtered_grad)
        return g, filtered_grad, grad_max
 
    def compute_pattern_and_mask_for_class(self, pois_model, detection_set, class_label, pattern_path):
        os.makedirs(pattern_path, exist_ok=True)
        pois_model.eval()
        #detection_set contains only the data for class "class_label"
        predicted_y = self.predict(pois_model, detection_set, prob_func=nn.Softmax(dim=1), x_transformer=None)
        ypred = torch.argmax(predicted_y, dim=1).detach().cpu().numpy()
        ind = np.where(ypred==class_label)[0]
        n = np.min([self.args.max_instances_for_variance, len(ind)]) #we need a max of n instances
        ind = ind[torch.sort(predicted_y[ind][:,class_label])[1][-n:].cpu().numpy()] #keep only the instances with the highest values
        if len(ind) >= self.args.min_instances_for_variance:
            img_set = torch.stack([detection_set[i][0] for i in ind])
            grad, grad_mask, grad_max = self.get_gradient_mask(pois_model, img_set, class_label)

            norm = get_dataset_normalization(self.args.dataset)
            proc_img_set = img_set.permute(0,2,3,1)*torch.tensor(norm.std) + torch.tensor(norm.mean)
            proc_img_set = torch.round(proc_img_set*255)
            img_var = self.compute_var(proc_img_set)
            
            img_mean = img_set.mean(dim=0)
            p_mask = self.otsu_binarize(grad_mask * (1.0-img_var) )
            #p_mask = grad_mask * self.otsu_binarize((1.0-img_var) )
            p_pattern = img_mean * p_mask
            #write images
            var_mask = self.otsu_binarize((1.0-img_var) )
            PIL.Image.fromarray((var_mask.detach().cpu().numpy()*255).astype(np.uint8), mode="L").save(pattern_path+f'/{class_label}_var_mask.png')
            PIL.Image.fromarray((img_var.detach().cpu().numpy()*255).astype(np.uint8), mode="L").save(pattern_path+f'/{class_label}_var.png')
            PIL.Image.fromarray((grad.detach().cpu().numpy()*255).astype(np.uint8), mode="L").save(pattern_path+f'/{class_label}_grad.png')
            PIL.Image.fromarray((grad_mask.detach().cpu().numpy()*255).astype(np.uint8), mode="L").save(pattern_path+f'/{class_label}_grad_mask.png')
            PIL.Image.fromarray((p_mask.detach().cpu().numpy()*255).astype(np.uint8), mode="L").save(pattern_path+f'/{class_label}_mask.png')
            p = (p_pattern.permute(1,2,0)*torch.tensor(norm.std) + torch.tensor(norm.mean))*255
            PIL.Image.fromarray(p.detach().cpu().numpy().astype(np.uint8), mode="RGB").save(pattern_path+f'/{class_label}_pattern.png')

            if torch.sum(p_mask) == 0:
                p_mask = None
        else:
            p_mask, p_pattern, grad_max = None, None, None
        return p_mask, p_pattern, grad_max
    
    def compute_pattern_and_mask(self, pois_model, poisoned_classes, detection_set, pattern_path):
        computed_pattern_file = pattern_path+'../computed_patterns.csv'
        if os.path.exists(computed_pattern_file):
            _m = f'File {computed_pattern_file} exists: reading patterns'
            logging.info(_m); #print(_m)

            trans = transforms.ToTensor()
            norm = get_dataset_normalization(self.args.dataset)
            pattern_set = {}
            with open(computed_pattern_file, 'r') as f:
                for l in f.readlines()[1:] :  #skip header
                    t = l.split(',')
                    pclass, pmodel_index = int(t[0]), int(t[1])
                    if pclass in poisoned_classes[pmodel_index]:
                        p_mask = trans(PIL.Image.open(pattern_path+f'/{pmodel_index}/{pclass}_mask.png'))
                        p_pattern = trans(PIL.Image.open(pattern_path+f'/{pmodel_index}/{pclass}_pattern.png'))
                        p_pattern = norm(p_pattern)
                        if pclass not in pattern_set:
                            pattern_set[pclass] = []
                        pattern_set[pclass].append( (pmodel_index, p_mask, p_pattern) )
            return pattern_set
            
        normal_trans = transforms.Compose([transforms.ToTensor(), get_dataset_normalization(self.args.dataset)])
        original_index = detection_set.wrapped_dataset.original_index_array
        original_trans = detection_set.wrap_img_transform
        detection_set.wrap_img_transform = normal_trans
        
        pattern_set = {}
        detect_labels = np.array(get_labels(detection_set))
        #grad_max_list = []
        for m_index, m in enumerate(pois_model):
            pclasses = []
            for cl in poisoned_classes[m_index]: #range(self.args.num_classes):
                cl_subset_inds = np.where(detect_labels==cl)[0]
                detection_set.wrapped_dataset.original_index_array = original_index
                detection_set.subset(cl_subset_inds)
                p_mask, p_pattern, grad_max = self.compute_pattern_and_mask_for_class(pois_model[m_index], detection_set, cl, pattern_path + f'/{m_index}/')
                #grad_max_list.append(grad_max)
                if p_mask is not None:
                    if cl not in pattern_set:
                        pattern_set[cl] = []
                    pattern_set[cl].append( (m_index, p_mask, p_pattern) )
                    pclasses.append(cl)
            logging.info(f'Model {m_index}: {len(pclasses)} patterns extracted for classes: {sorted(pclasses)}')

        detection_set.wrapped_dataset.original_index_array = original_index
        detection_set.wrap_img_transform = original_trans

        with open(computed_pattern_file, 'w') as f:
            f.write('class,model\n')
            for k in sorted(pattern_set.keys()):
                t = sorted([ m_index for (m_index, p_mask, p_pattern) in pattern_set[k] ])
                for i in t:
                    f.write(f'{k},{i}\n')

        return pattern_set

    def test_patterns(self, pois_model, pois_model_acc, pattern_set, detection_set, pattern_path):
        attack_pattern_file = pattern_path + '../attack_patterns.csv'
        asr_file = pattern_path + '../asr_patterns.csv'

        if os.path.exists(attack_pattern_file):
            logging.info(f'File {attack_pattern_file} exists: reading attack patterns')
            attack_pattern_set = {}
            n_patterns = 0
            with open(attack_pattern_file, 'r') as f:
                for l in f.readlines()[1:] :  #skip header
                    t = l.split(',')
                    pclass, pmodel_index = int(t[0]), int(t[1])
                    if pclass not in attack_pattern_set:
                        attack_pattern_set[pclass] = []
                    for (pmi, p_mask, p_pattern) in pattern_set[pclass]:
                        if pmi == pmodel_index:
                            attack_pattern_set[pclass].append( (pmodel_index, p_mask, p_pattern) )
                            n_patterns += 1
            logging.info(f"Number of patterns detected: {n_patterns} for classes ({sorted([int(i) for i in attack_pattern_set.keys()])})")
            return attack_pattern_set

        normal_trans = transforms.Compose([transforms.ToTensor(), get_dataset_normalization(self.args.dataset)])
        original_index = detection_set.wrapped_dataset.original_index_array
        original_trans = detection_set.wrap_img_transform
        detection_set.wrap_img_transform = normal_trans

        labels = np.array(get_labels(detection_set))
        subset = self.strat_subset(labels, 0.1)  #make the test on 10% of the database
        sublabels = labels[subset]
        logging.info(f'Testing with a subset of 10% ({len(subset)}/{len(labels)})')

        asr_tab= []
        attack_pattern_set = {}
        asr_pattern_map = {}
        for target_class in sorted(pattern_set.keys()):
            testset = subset[sublabels != target_class] #remove instances from the target_class
            detection_set.wrapped_dataset.original_index_array = original_index
            detection_set.subset(testset)
            for p_model, p_mask, p_pattern in pattern_set[target_class]:
                pattern_adder = PatternAdder(p_mask, p_pattern, alpha=1).to(self.device, non_blocking=self.args.non_blocking)
                predicted_y = self.predict(pois_model[p_model], detection_set, prob_func=nn.Softmax(dim=1), x_transformer=pattern_adder)
                predicted_y = torch.argmax(predicted_y, dim=1).detach().cpu().numpy()
                asr = np.sum(predicted_y==target_class) / float(len(predicted_y))
                asr_tab.append( [p_model, target_class, asr] )
                if asr >= self.args.pattern_min_asr:
                    if target_class not in attack_pattern_set:
                        attack_pattern_set[target_class] = []
                        asr_pattern_map[target_class] = []
                    attack_pattern_set[target_class].append( (p_model, p_mask, p_pattern) )
                    asr_pattern_map[target_class].append(asr)
                logging.info(f"Testing model {p_model} pattern for target class {target_class}, ASR: {int(np.round(asr*100.0))}%")

        logging.info(f"Number of patterns detected: {len(attack_pattern_set)} ({sorted([int(i) for i in attack_pattern_set.keys()])})")

        n_models = len(pois_model)
        #n_thr = n_models / 2.0
        n_thr = int(self.args.ensemble_detect_min_thr)
        filtered_attack_pattern_set = {}
        for target_class in sorted(attack_pattern_set.keys()):
            patts = attack_pattern_set[target_class]
            n_votes = len(patts)
            logging.info(f'Class {target_class} detected by {n_votes}/{n_models} models'+(f':thr reached (>={n_thr})' if (n_votes >= n_thr) else ''))
            if n_votes >= n_thr:
                #keep only the pattern with both the lowest acc and highest asr
                p_model_indexes = [p[0] for p in patts]
                models_acc = pois_model_acc[p_model_indexes, 0] #get the train accuracies of the models
                asr_vals = np.array(asr_pattern_map[target_class])
                scores = (1. - models_acc)*0.6 + asr_vals*0.4
                ind = np.argmax(scores)
                filtered_attack_pattern_set[target_class] = [patts[ind]]
                logging.info(f'Class {target_class}: Keeping model {patts[ind][0]} with train acc {(100*models_acc[ind])}% and asr {(100*asr_vals[ind])}%  (score: {scores[ind]})')

        attack_pattern_set = filtered_attack_pattern_set
        logging.info(f"After filtering, number of patterns detected: {len(attack_pattern_set)} ({sorted([int(i) for i in attack_pattern_set.keys()])})")

        with open(attack_pattern_file, 'w') as f:
            f.write('class,model\n')
            for cl in sorted(attack_pattern_set.keys()):
                for i in sorted([k for k,_,_ in attack_pattern_set[cl]]):
                    f.write(f'{cl},{i}\n')

        with open(asr_file, 'w') as f:
            f.write('model,class,asr\n')
            for p_model, target_class, asr in asr_tab:
                f.write(f'{p_model},{target_class},{np.round(asr,4)}\n')

        detection_set.wrapped_dataset.original_index_array = original_index
        detection_set.wrap_img_transform = original_trans
        return attack_pattern_set
    
    def predict(self, model, dataset, prob_func=None, x_transformer=None):
        model.eval()
        if type(dataset) != DataLoader:
            loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False,
                                    pin_memory=self.args.pin_memory, num_workers=self.args.num_workers, )
        else:
            loader = dataset
        predicted_y = [] if type(model) is not EnsembleModel else [ [] for i in range(len(model)) ]
        for x, *additional_info in loader:
            x = x.to(self.device, non_blocking=self.args.non_blocking)

            if type(model) is EnsembleModel:
                pred = model(x, transf=x_transformer, prob_func=prob_func, no_grad=True)
                for i, p  in enumerate(pred):
                    predicted_y[i].append(p.detach())
            else:
                if x_transformer is not None:
                    x = x_transformer(x)
                with torch.no_grad():
                    pred = model(x)
                    if prob_func is not None:
                        pred = prob_func(pred)
                    predicted_y.append(pred.detach())
        
        if type(model) is EnsembleModel:
            new_pred = []
            for p in predicted_y:
                if len(p) == 0:
                    new_pred.append(torch.tensor([]).to(self.device))
                else:
                    new_pred.append(torch.concatenate(p, dim=0))
            predicted_y = new_pred
        else:
            predicted_y = torch.tensor([]).to(self.device) if (len(predicted_y) == 0) else torch.concatenate(predicted_y, dim=0)
        return predicted_y
        

    def inject_poison_for_training(self, x, pattern, mask, alpha_min=0.2, alpha_max=1.0):
        x_shape= x.shape
        n = x_shape[0]
        alpha = torch.tensor([alpha_min]*n)
        if (alpha_min<alpha_max):
            alpha = torch.tensor(np.random.uniform(alpha_min, alpha_max, len(x)), dtype=torch.float32)
        alpha = alpha.unsqueeze(1).to(x.device.type)
        x = (x*(1-mask)).view(n, -1)  + ( (1.-alpha)*(x*(mask)).view(n, -1)) + (alpha *(pattern*mask).unsqueeze(3).tile(n).permute(3,0,1,2).view(n,-1))
        x = x.view(x_shape)
        return x
    
    def compute_binary_metrics(self, predicted_y, pred_y_label, y_true, loss_func=nn.BCELoss()):
        assert(predicted_y is None or type(predicted_y)==np.ndarray)
        assert(type(pred_y_label)==np.ndarray)
        assert(type(y_true)==np.ndarray)
        scores = {}
        scores['tp'] = tp = int(np.sum(pred_y_label * y_true))
        scores['tn'] = tn = int(np.sum( (1-pred_y_label) * (1-y_true) ))
        scores['fp'] = fp = int(np.sum(pred_y_label * (1-y_true)))
        scores['fn'] = fn = int(np.sum((1-pred_y_label) * y_true))
        scores['loss'] = loss = np.nan if (loss_func is None or predicted_y is None) else loss_func(torch.FloatTensor(predicted_y), torch.FloatTensor(y_true)).item()
        scores['acc'] = acc = np.min([1.0, (tp+tn)/(tp+fn+tn+fp+1e-8)])
        scores['tpr'] = tpr = np.min([1.0, tp/(tp+fn+1e-8)])
        scores['fpr'] = fpr = np.min([1.0, fp/(tn+fp+1e-8)])
        scores['precision'] = precision = np.min([1.0, tp/(tp+fp+1e-8)])
        scores['recall'] = recall = np.min([1.0, tp/(tp+fn+1e-8)])
        scores['f1'] = f1 = np.min([1.0, 1.0/(0.5 * ((1./(precision+1e-8))+(1./(recall+1e-8))) ) ])
        scores_str =  ", ".join([f'{k}:{np.round(scores[k], 2)}' for k in ['loss', 'acc', 'tpr', 'fpr', 'f1', 'tp', 'fp', 'tn', 'fn'] ])
        f'loss:{np.round(loss,2)}, acc:{acc}, tpr:{tpr}, fpr:{fpr}, f1:{f1}, tp:{tp}, fp:{fp}, tn:{tn}, fn:{fn}'
        return scores, scores_str

    def detect_poisoned_samples(self, model, dataset, x_transformer=None):
        predicted_y = self.predict(model, dataset, prob_func=None, x_transformer=x_transformer)
        if type(model) is EnsembleModel:
            pred_y_label = torch.zeros(len(predicted_y[0])).to(self.args.device)
            avg_predicted = torch.zeros(predicted_y[0].shape).to(self.args.device)
            for i in range(len(model)):
                pred_y_label += torch.where(predicted_y[i] >= 0.5, 1, 0).detach().squeeze()
                avg_predicted += predicted_y[i]
            pred_y_label = torch.where(pred_y_label >= 0.5, 1, 0)
            poisoned_indices = torch.where(pred_y_label==1)[0].cpu().numpy()
            avg_predicted /= len(model)
            return poisoned_indices, avg_predicted.detach().cpu().squeeze().numpy(), pred_y_label.detach().cpu().numpy()
        else:
            pred_y_label = torch.where(predicted_y >= 0.5, 1, 0).detach().squeeze()
            poisoned_indices = torch.where(pred_y_label==1)[0].cpu().numpy()
            return poisoned_indices, predicted_y.detach().cpu().squeeze().numpy(), pred_y_label.detach().cpu().numpy()

    def train_pattern_detectors(self, pois_model, pattern_set, detection_set, pattern_path):
        detector_set_filename = pattern_path + '/../detectors.pt'
        normal_trans = transforms.Compose([transforms.ToTensor(), get_dataset_normalization(self.args.dataset)])
        original_index = detection_set.wrapped_dataset.original_index_array
        original_trans = detection_set.wrap_img_transform
        detection_set.wrap_img_transform = normal_trans

        if os.path.exists(detector_set_filename):
            logging.info(f'File {detector_set_filename} exists: reading detector models')
            flat_dict = torch.load(detector_set_filename)
            detector_set = {}
            binary_models = EnsembleModel(self.args.input_height, self.args.input_width, self.args.num_classes, models=[])
            for k, (mod_indexes, em) in flat_dict.items():
                for m in em:
                    b = SimpleClassifierModel(self.args.input_height, self.args.input_width, classes=1, use_bn=False, dropout_perc=0.5, output_prob=True, use_sigmoid=True)
                    b.load_state_dict(m)
                    binary_models.add(b)
                detector_set[k] = (mod_indexes, binary_models)
            binary_models = self.move_to_device(binary_models)
            binary_models = binary_models.eval()
            logging.info(f"Number of detectors loaded: {len(detector_set)} ({sorted([int(i) for i in detector_set.keys()])})")
            return detector_set

        train_set_inds = np.arange(len(detection_set))
        train_labels = np.array(get_labels(detection_set))
        if self.args.detector_train_perc < 1.0:
            subratio = self.args.detector_train_perc
            subset = self.strat_subset(train_labels, subratio) 
            sublabels = train_labels[subset]
            logging.info(f'Learning detector with a subset of {int(subratio*100)}% ({len(subset)}/{len(train_labels)})')
            train_set_inds = subset
            train_labels = sublabels
        
        #test_set is the same as the training set
        test_set = dataset_wrapper_with_transform(detection_set.wrapped_dataset.copy(), normal_trans, detection_set.wrap_label_transform)
        test_set_inds = train_set_inds
        test_labels = train_labels

        detector_set = {}
        for target_class in sorted(pattern_set.keys()):
            logging.info(f'Training detector for class {target_class}')

            optimizers_list = []
            data_transf_list, pattern_adder_datatransf_list = [],  []
            model_indexes = []
            binary_models = EnsembleModel(self.args.input_height, self.args.input_width, self.args.num_classes, models=[])
            for p_model_index, p_mask, p_pattern in pattern_set[target_class]:
                model_indexes.append(p_model_index)
                b = SimpleClassifierModel(self.args.input_height, self.args.input_width, classes=1, use_bn=False, dropout_perc=0.5, output_prob=True, use_sigmoid=True)
                binary_models.add(b)
                optimizer, scheduler = argparser_opt_scheduler(b, self.args)
                optimizers_list.append( (optimizer, scheduler) )
                g = GradTransf(pois_model[p_model_index], target_class)
                data_transf_list.append( g )
                p = PatternAdder(p_mask, p_pattern, (0.2, 1.0)).to(self.device, non_blocking=self.args.non_blocking)
                pattern_adder_datatransf_list.append( (p, g) ) #pattern adder followed by grad transf
            
            binary_models = self.move_to_device(binary_models)
            binary_models = binary_models.train()
            
            t_subset = train_set_inds[train_labels != target_class] #remove instances from the target_class
            detection_set.wrapped_dataset.original_index_array = original_index
            detection_set.subset(t_subset)
            detect_loader = DataLoader(detection_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True,
                                   pin_memory=self.args.pin_memory, num_workers=self.args.num_workers, )

            t_subset = test_set_inds[test_labels == target_class]
            test_set.wrapped_dataset.original_index_array = original_index
            test_set.subset(t_subset)
            test_set_inds_new = test_set.wrapped_dataset.original_index_array
            test_set_true_poison = test_set.poison_indicator[t_subset]
            
            predicted_y = None
            validation_random_seed = 10

            loss = nn.BCELoss().to(self.device, non_blocking=self.args.non_blocking)
            train_test_set = None
            best_loss, best_model_params = None, None
            for epoch in range(self.args.detector_max_epochs):
                predicted_y, y_true = [], []
                binary_models.train()
                for p, g in pattern_adder_datatransf_list:
                    p.set_random_gen(None) #default random generator

                train_test_set_loader_iter = None
                if train_test_set is not None and len(train_test_set)>=self.args.batch_size:
                    train_test_set_loader = DataLoader(train_test_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True,
                                                    pin_memory=self.args.pin_memory, num_workers=self.args.num_workers, )
                    train_test_set_loader_iter = iter(train_test_set_loader)
                detect_loader_iter = iter(detect_loader)
                full_loss = [0]*len(binary_models)
                n_data = 0
                while True:
                    l_y_true = []
                    try:
                        x, target, *additional_info = next(detect_loader_iter)
                        x = x.to(self.device, non_blocking=self.args.non_blocking)
                        l_y_true += [0]*x.shape[0]
                        #compute the gradient of the clean image and predict
                        pred = binary_models(x, transf=data_transf_list)

                        #inject pattern with random alpha
                        pred_pois = binary_models(x, transf=pattern_adder_datatransf_list)
                        l_y_true += [1]*x.shape[0]
                        for i in range(len(pred)):
                            pred[i] = torch.cat([pred[i], pred_pois[i]], dim=0)
                        y = torch.cat([torch.zeros(x.shape[0],1), torch.ones(x.shape[0],1)]).to(self.device, non_blocking=self.args.non_blocking)
                    except StopIteration:
                        break

                    if train_test_set_loader_iter is not None:
                        try:
                            x, pois_target, *oth = next(train_test_set_loader_iter)
                            x = x.to(self.device, non_blocking=self.args.non_blocking)
                            l_y_true += pois_target.tolist()
                            #compute the gradient of the cimage and predict
                            pred2 = binary_models(x, transf=data_transf_list)
                            for i in range(len(pred)):
                                pred[i] = torch.cat([pred[i], pred2[i]], dim=0)
                            pois_target = pois_target.to(self.device, non_blocking=self.args.non_blocking)
                            pois_target = pois_target.unsqueeze(1)
                            y = torch.cat((y, pois_target))
                        except StopIteration:
                            train_test_set_loader_iter = None
                            break
                    
                    y_true += l_y_true
                    if (predicted_y is None) or len(predicted_y)==0:
                        predicted_y = [ [p.detach().squeeze()] for p in pred]
                    else:
                        for i in range(len(pred)):
                            predicted_y[i].append(pred[i].detach().squeeze())
                    
                    n_data += pred[i].shape[0]
                    for i, (optimizer, scheduler) in enumerate(optimizers_list):
                        l = loss(pred[i], y)
                        full_loss[i] += l.item() * pred[i].shape[0]
                        optimizer.zero_grad()
                        l.backward()
                        optimizer.step()
                full_loss = [f/n_data for f in full_loss]
                binary_models.eval()
                logging.info(f'Training detector for class {target_class} epoch {epoch+1}:')
                pred_y_label = []
                global_y_pred = None
                curr_loss = []
                for i in range(len(predicted_y)):
                    predicted_y[i] = torch.cat(predicted_y[i]).detach().cpu().numpy()
                    pred_y_label.append( np.where(predicted_y[i]>=0.5, 1, 0) )
                    global_y_pred = pred_y_label[i] if global_y_pred is None else (global_y_pred + pred_y_label[i])
                    scores, scores_str = self.compute_binary_metrics(predicted_y[i], pred_y_label[i], np.array(y_true), loss_func=nn.BCELoss())
                    logging.info(f'  Perf on training (model {i}): {scores_str}')
                    curr_loss.append(scores['loss'])
                
                global_y_pred = np.where(global_y_pred>=0.5, 1, 0) #if at least one detector has identified
                scores, scores_str = self.compute_binary_metrics(None, global_y_pred, np.array(y_true), loss_func=None)
                logging.info(f'  Perf on training (global): {scores_str}')
                global_acc = scores['acc']

                if epoch>=5 and epoch<12:

                    predicted_y = self.predict(binary_models, test_set, x_transformer=data_transf_list)
                    g_predicted_y = None
                    for i in range(len(predicted_y)):
                        predicted_y[i] = predicted_y[i].detach().cpu().numpy()
                        res = np.where(predicted_y[i]>=0.5, 1, 0)
                        g_predicted_y = res if g_predicted_y is None else (g_predicted_y + res)
                    t_pois_inds = np.where(g_predicted_y>=0.5)[0]
                    t_neg_pois = np.where(g_predicted_y<0.5)[0]
                    labind = {i:1 if i<len(t_pois_inds) else 0 for i in range(len(t_pois_inds)+len(t_neg_pois))}

                    if len(labind)>0:
                        if train_test_set is None:
                            train_test_set = dataset_wrapper_with_transform(test_set.wrapped_dataset.copy(), normal_trans, test_set.wrap_label_transform)
                        train_test_set.wrapped_dataset.original_index_array = test_set_inds_new
                        train_test_set.subset(np.concatenate((t_pois_inds, t_neg_pois), axis=0))
                        train_test_set = relabeled_dataset_wrapper_with_transform(train_test_set.wrapped_dataset.copy(), normal_trans, train_test_set.wrap_label_transform, labind)
                
                poisoned_indices, predicted_y, pred_y_label = self.detect_poisoned_samples(binary_models, test_set, x_transformer=data_transf_list)
                scores, scores_str = self.compute_binary_metrics(predicted_y, pred_y_label, test_set_true_poison, loss_func=None)#nn.BCELoss())
                logging.info(f'  Perf on test: {scores_str}')

                for i, (optimizer, scheduler) in enumerate(optimizers_list):
                    scheduler.step(curr_loss[i])
            
            if global_acc<0.75:
                logging.info(f'Accuracy ({global_acc}) is too bad to detect poisoned instances for class {target_class} ')
            else:
                detector_set[target_class] = (model_indexes, binary_models)
        
        flat_dict = {int(k): (mod_inds, [m.state_dict() for m in em]) for k, (mod_inds, em) in detector_set.items()}
        torch.save(flat_dict, detector_set_filename)

        detection_set.wrapped_dataset.original_index_array = original_index
        detection_set.wrap_img_transform = original_trans
        return detector_set
    
    def detect_all_poisons(self, pois_model, detector_set, dataset, pattern_path, dataset_type='train', true_labels = None, true_poisoned_flags = None):
        poison_indices_filename = pattern_path + f'/../{dataset_type}_poison_indices.csv'
        perf_filename = pattern_path + f'/../{dataset_type}_perf_poison_detection.csv'
        normal_trans = transforms.Compose([transforms.ToTensor(), get_dataset_normalization(self.args.dataset)])
        original_index = dataset.wrapped_dataset.original_index_array
        original_trans = dataset.wrap_img_transform
        dataset.wrap_img_transform = normal_trans

        if true_poisoned_flags is not None:
            perf_keys = ['loss', 'acc', 'tpr', 'fpr', 'f1', 'tp', 'fp', 'tn', 'fn']
            f = open(perf_filename, 'w')
            f.write('class,'+",".join(perf_keys)+'\n')

        pred_flags = np.array([0]*len(dataset))
        pred_y = np.array([0.]*len(dataset))
        labels = np.array(get_labels(dataset))
        for cl, (model_indexes, model) in detector_set.items():
            logging.info(f'Detecting poison in class {cl}  for {dataset_type} dataset...')
            cl_indices = np.where(labels==cl)[0]
            dataset.wrapped_dataset.original_index_array = original_index
            dataset.subset(cl_indices)
            if type(pois_model) is EnsembleModel:
                data_transf = [GradTransf(pois_model[i], cl) for i in model_indexes]
            else:
                data_transf = GradTransf(pois_model, cl)
            poisoned_indices, predicted_y, pred_y_label = self.detect_poisoned_samples(model, dataset, x_transformer=data_transf)
            poisoned_indices = cl_indices[poisoned_indices] #to original indices
            pred_y[cl_indices] = predicted_y
            pred_flags[poisoned_indices] = 1
            if true_poisoned_flags is not None:
                true_flags = true_poisoned_flags[cl_indices]
                scores, scores_str = self.compute_binary_metrics(predicted_y, pred_y_label, true_flags, loss_func=nn.BCELoss())
                logging.info(f'Perf for class {cl}: {scores_str}')
                scores_str = f'{cl},' + ",".join([f'{k}:{np.round(scores[k], 2)}' for k in perf_keys ])
                f.write(scores_str+'\n')
        
        if true_poisoned_flags is not None:
            scores, scores_str = self.compute_binary_metrics(pred_y, pred_flags, true_poisoned_flags, loss_func=nn.BCELoss())
            logging.info(f'Overall Perf : {scores_str}')
            scores_str = f'ALL,' + ",".join([f'{k}:{np.round(scores[k], 2)}' for k in perf_keys ])
            f.write(scores_str+'\n')
            f.close()
        
        with open(poison_indices_filename, 'w') as f:
            is_poison_true = [str(int(i)) for i in true_poisoned_flags] if true_poisoned_flags is not None else ['']*len(labels)
            t_labels = ['']*len(labels) if true_labels is None else [str(i) for i in true_labels]
            f.write('index,class,true_class, is_poison_true,is_poison_pred,score\n')
            for a in zip(range(len(labels)), labels, t_labels, is_poison_true, pred_flags, pred_y):
                f.write(','.join([str(i) for i in a]) +'\n')

        dataset.wrapped_dataset.original_index_array = original_index
        dataset.wrap_img_transform = original_trans
        return pred_flags, pred_y


    def mitigation(self):
        if type(self.args.no_retrain) == str:
            self.args.no_retrain = (self.args.no_retrain.lower() == "true")
        self.set_devices()
        fix_random(self.args.random_seed)

        self.args.pois_train_perc = float(self.args.pois_train_perc)
        self.args.detector_train_perc = float(self.args.detector_train_perc)

        pois_model_path = self.args.save_path+'/pois_model/'
        pattern_path = self.args.save_path+'/patterns/'
        os.makedirs(pois_model_path, exist_ok=True)
        os.makedirs(pattern_path, exist_ok=True)

        #get the dataset
        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        normal_trans = transforms.Compose([transforms.ToTensor(), get_dataset_normalization(self.args.dataset)])
        train_dataset = self.result['bd_train'].wrapped_dataset
        data_set_without_tran = train_dataset
        data_set_o = self.result['bd_train']
        data_set_o.wrapped_dataset = data_set_without_tran
        data_set_o.wrap_img_transform = train_tran
        data_set_o.wrapped_dataset.getitem_all = True  #to have img, label, original index, original, label

        bd_train_len = len(data_set_o)

        train_labels = np.array(get_labels(train_dataset))
        if self.args.pois_train_perc<1.0:
            pois_train_indexes, pois_detect_indexes = \
                self.stage_1_split_train_detect(train_labels, self.args.pois_train_perc, pois_model_path)
        else:
            pois_train_indexes = pois_detect_indexes = np.arange(len(train_labels))  #use the full dataset for training
        
        pois_model, pois_model_acc = self.stage_1_learn_poison_model(self.result['bd_train'], 
                                                                     train_labels,
                                                                     pois_train_indexes, 
                                                                     pois_detect_indexes,
                                                                     self.result['clean_test'], 
                                                                     self.result['bd_test'], 
                                                                     pois_model_path)
        
        

        logging.info("Stage 2: Detecting poisoned class ...")
        original_index = data_set_o.wrapped_dataset.original_index_array
        data_set_o.subset(pois_detect_indexes)

        poisoned_classes = self.detect_poisoned_classes(pois_model, data_set_o, save_path=self.args.save_path+'/')

        data_set_o.wrapped_dataset.original_index_array = original_index
        data_set_o.wrap_img_transform = train_tran

        pattern_set = {}
        if len(poisoned_classes) > 0:
            logging.info("Stage 3: Computing patterns ...")

            original_index = data_set_o.wrapped_dataset.original_index_array
            data_set_o.subset(pois_detect_indexes)

            pattern_set = self.compute_pattern_and_mask(pois_model, poisoned_classes, data_set_o, pattern_path)

            data_set_o.wrapped_dataset.original_index_array = original_index
            data_set_o.wrap_img_transform = train_tran

        attack_pattern_set = {}
        if len(pattern_set)>0:
            logging.info("Stage 4: Testing patterns ...")
            original_index = data_set_o.wrapped_dataset.original_index_array
            data_set_o.subset(pois_detect_indexes)

            attack_pattern_set = self.test_patterns(pois_model, pois_model_acc, pattern_set, data_set_o, pattern_path)

            data_set_o.wrapped_dataset.original_index_array = original_index
            data_set_o.wrap_img_transform = train_tran

        detector_set = {}
        if len(attack_pattern_set) >0:
            logging.info("Stage 5: Training the pattern detectors ...")
            original_index = data_set_o.wrapped_dataset.original_index_array
            #data_set_o.subset(pois_detect_indexes) #use the full set to train the detectors

            detector_set = self.train_pattern_detectors(pois_model, attack_pattern_set, data_set_o, pattern_path)

            data_set_o.wrapped_dataset.original_index_array = original_index
            data_set_o.wrap_img_transform = train_tran

        pred_flags = np.array([0]*len(data_set_o))
        if len(detector_set) > 0:
            logging.info("Stage 6: Detecting the pattern in the dataset ...")
            original_index = data_set_o.wrapped_dataset.original_index_array
            
            t = np.array([(i[4], i[3]) for i in data_set_o])
            true_labels, poison_indicator = t[:,0], t[:,1]
            pred_flags, pred_y = self.detect_all_poisons(pois_model, detector_set, data_set_o, pattern_path, dataset_type='train', true_labels=true_labels, true_poisoned_flags = poison_indicator)
            
            clean_test_dataset = dataset_wrapper_with_transform(prepro_cls_DatasetBD_v2(self.result['clean_test'].wrapped_dataset), self.result['clean_test'].wrap_img_transform, self.result['clean_test'].wrap_label_transform)
            clean_test_pred_flags, clean_test_pred_y =self.detect_all_poisons(pois_model, detector_set, clean_test_dataset, pattern_path, dataset_type='clean_test', true_poisoned_flags = np.array([0]*len(clean_test_dataset)))
            
            bd_test_dataset = self.result['bd_test']
            test_trans = bd_test_dataset.wrap_img_transform
            bd_test_dataset.wrap_img_transform = None
            t = np.array([(i[4], i[3]) for i in bd_test_dataset])
            true_labels, poison_indicator = t[:,0], t[:,1]
            bd_test_dataset.wrap_img_transform = test_trans
            bd_test_pred_flags, bd_test_pred_y = self.detect_all_poisons(pois_model, detector_set, bd_test_dataset, pattern_path, dataset_type='bd_test', true_labels=true_labels, true_poisoned_flags = poison_indicator)

            data_set_o.wrapped_dataset.original_index_array = original_index
            data_set_o.wrap_img_transform = train_tran

        data_set_o.wrapped_dataset.original_index_array = original_index
        data_set_o.wrap_img_transform = train_tran
        self.result['dataset'] = data_set_o
        model = generate_cls_model(self.args.model, self.args.num_classes)
        if (np.sum(pred_flags) == 0) or (self.args.no_retrain):
            if self.args.no_retrain:
                logging.info("no_retrain is set to True => no retraining (detection mode only)")
            model.load_state_dict(self.result['model'])
            model.eval()
            model = model.to(self.args.device)
            max_epochs = 1 # only one epoch : just to to have the test result
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0,momentum=0, weight_decay=0) #set the lr to 0 : we don't want to learn
            scheduler = None
        else:
            logging.info("Stage 7: Retraining with the clean training dataset ...")
            #filtered data
            if "," in self.device:
                model = torch.nn.DataParallel(
                    model,
                    device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
                )
                self.args.device = f'cuda:{model.device_ids[0]}'
                model.to(self.args.device)
            else:
                model.to(self.args.device)
            max_epochs = self.args.epochs
            optimizer, scheduler = argparser_opt_scheduler(model, self.args)
            
        data_set_o.wrapped_dataset.original_index_array = original_index
        data_set_o.wrap_img_transform = train_tran
        data_set_o.wrapped_dataset.getitem_all = True        
            
        if self.args.retrain_pois_method == 'relabel_with_det_pois_model':
            logging.info('Retraining with relabeling the poison instances with the poison detection model')
            #relabel the poison instances using the second label with the highest score
            labind = {i:pois_model(data_set_o[i][0].unsqueeze(0).to(self.device))[0].sort()[1][0][-2].item() for i,v in enumerate(pred_flags) if v==1}
            data_set_o2 = relabeled_dataset_wrapper_with_transform(data_set_o.wrapped_dataset.copy(), train_tran, data_set_o.wrap_label_transform, labind)
        elif self.args.retrain_pois_method == 'relabel_with_pois_model':
            logging.info('Retraining with relabeling the poison instances with the poison model')
            #relabel the poison instances using the second label with the highest score
            original_pois_model = generate_cls_model(self.args.model, self.args.num_classes)
            original_pois_model.load_state_dict(self.result['model'])
            original_pois_model.eval()
            original_pois_model = original_pois_model.to(self.device)
            labind = {i:original_pois_model(data_set_o[i][0].unsqueeze(0).to(self.device)).sort()[1][0][-2].item() for i,v in enumerate(pred_flags) if v==1}
            data_set_o2 = relabeled_dataset_wrapper_with_transform(data_set_o.wrapped_dataset.copy(), train_tran, data_set_o.wrap_label_transform, labind)
        else:
            #self.args.retrain_pois_method == 'suppress':
            logging.info('Retraining after suppressing the poison instances')
            data_set_o2 = dataset_wrapper_with_transform(data_set_o.wrapped_dataset.copy(), train_tran, data_set_o.wrap_label_transform)
            data_set_o2.subset([i for i,v in enumerate(pred_flags) if v==0])

        data_set_o.subset([i for i,v in enumerate(pred_flags) if v==0]) #clear the detected poisons

        data_set_o2.wrapped_dataset.getitem_all = True
        data_loader_sie = torch.utils.data.DataLoader(data_set_o2, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, drop_last=True)
        
        self.set_trainer(model)
        criterion = argparser_criterion(self.args)

        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=self.args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=self.args.pin_memory)


        if self.args.no_retrain == False:
            self.trainer.train_with_test_each_epoch_on_mix(
                data_loader_sie,
                data_clean_loader,
                data_bd_loader,
                max_epochs,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self.args.device,
                frequency_save=self.args.frequency_save,
                save_folder_path=self.args.save_path,
                save_prefix='vbd',
                amp=self.args.amp,
                prefetch=self.args.prefetch,
                prefetch_transform_attr_name="ori_image_transform_in_loading", # since we use the preprocess_bd_dataset
                non_blocking=self.args.non_blocking,
            )
        
        result = {}
        result['model'] = model
        result['dataset'] = data_set_o  #cleaned train dataset
        save_defense_result(
            model_name=self.args.model,
            num_classes=self.args.num_classes,
            model=model.cpu().state_dict(),
            save_path=self.args.save_path,
        )
        return result     

    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result






if __name__ == '__main__':
    import matplotlib
    matplotlib.use('AGG')

    parser = argparse.ArgumentParser(description=sys.argv[0])
    VarianceBasedDefense.add_arguments(parser)
    args = parser.parse_args()
    defense_method = VarianceBasedDefense(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_vbd_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_vbd_badnet'
    result = defense_method.defense(args.result_file)

