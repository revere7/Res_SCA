# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:25:14 2019

@author: sun
"""
import os
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import copy
import logging
import random
import scipy.io as sio
#import matplotlib.pyplot as plt
import time


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.nn.parameter import Parameter

#from filters import all_normalized_hpf_list
#from srm_filter_kernel import all_normalized_hpf_list

IMAGE_SIZE = 256
BATCH_SIZE = 32 // 2
EPOCHS = 280

LR = 0.002
WEIGHT_DECAY = 5e-4


TRAIN_FILE_COUNT = 14000
TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1
DECAY_EPOCH = [130, 230]
TMP = 230

OUTPUT_PATH = Path(__file__).stem

SRM_npy = np.load('./SRM_Kernels.npy')


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
        

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=5):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        #mip = max(8, inp // reduction)
        mip = max(15, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
        

class SRM_conv2d(nn.Module):
    def __init__(self, stride=1, padding=2):
        super(SRM_conv2d, self).__init__()
        self.in_channels = 1
        self.out_channels = 30
        self.kernel_size = (5, 5)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.dilation = (1, 1)
        self.transpose = False
        self.output_padding = (0,)
        self.groups = 1
        self.weight = Parameter(torch.Tensor(30, 1, 5, 5), \
                                requires_grad=True)
        self.bias = Parameter(torch.Tensor(30), \
                              requires_grad=True)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, \
                        self.stride, self.padding, self.dilation, \
                        self.groups)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 with_bn=True, with_acti=True):
        super(ConvBlock, self).__init__()
        self.with_bias = False if with_bn else True 

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                              padding=(kernel_size-1)//2, bias=self.with_bias)
        
        self.norm = nn.BatchNorm2d(out_channels) if with_bn else nn.Identity()
        self.relu = nn.ReLU() if with_acti else nn.Identity()
        self.with_bn = with_bn
        
        self.reset_parameters()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        if self.with_bn:
            self.norm.reset_parameters()
        else:
            self.conv.bias.data.fill_(0.2)


class ResBlock(nn.Module):
    def __init__(self, c1, c2, poolsize=3, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(c1, c1, 3, 1, with_bn=True, with_acti=True)
        self.conv2 = ConvBlock(c1, c2, poolsize, stride, with_bn=True, with_acti=False)
        if stride>1:
            self.idt_map = ConvBlock(c1, c2, poolsize, stride, with_bn=True, with_acti=False)
        else:
            self.idt_map = nn.Identity()

    def forward(self, x):
        shortcut = self.idt_map(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x + shortcut)
        return x


class ResBlock2(nn.Module):
    def __init__(self, in_c, c1, c2, poolsize=2, stride=2,ks=None):
        super(ResBlock2, self).__init__()
        self.pool = nn.AvgPool2d(poolsize, stride, padding=(poolsize-1)//2)
        self.idt_map = ConvBlock(in_c, c2, 1, 1, with_bn=True, with_acti=False)
        kernel_size = ks if ks else 3
        self.conv1 = ConvBlock(in_c, c1, 1, 1, with_bn=True, with_acti=True)
        self.conv2 = ConvBlock(c1, c1, kernel_size, stride, with_bn=True, with_acti=True)
        self.conv3 = ConvBlock(c1, c2, 1, 1, with_bn=True, with_acti=False)

    def forward(self, x):
        shortcut = self.idt_map(self.pool(x))
        x = self.conv1(x)
        x = self.conv2(x) 
        x = self.conv3(x)
        x = F.relu(x + shortcut)
        return x  


class Net(nn.Module):
    def __init__(self, threshold=31):
        super(Net, self).__init__()
        self.preconv = SRM_conv2d()
        self.preconv_weight = self.preconv.weight
        self.preconv_bias = self.preconv.bias
        
        self.coord = CoordAtt(30, 30, reduction=5)
		
        self.TLU = nn.Hardtanh(-threshold, threshold, True)
        self.conv_a = ConvBlock(30, 30)
        self.conv_b = ConvBlock(30, 30) 
		
        self.conv_a_ = ConvBlock(30, 30)
        self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.conv_b_ = ConvBlock(30, 30) 
		
        self.conv_c = ConvBlock(30, 30)
        self.block1 = ResBlock(30, 30)
        self.block2 = ResBlock(30, 30)
        self.block3 = ResBlock(30, 60, poolsize=3, stride=2)
        self.block4 = ResBlock2(60, 30, 64, 2, 2)
        self.block5 = ResBlock2(64, 32, 128, 3, 2)
        self.block6 = ResBlock2(128, 64, 256, 3, 3)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 2)
        self.reset_parameters()

    def forward(self, x, y):	    
        x = x.float()
        x = self.preconv(x)
        x = self.TLU(x)
        x = self.conv_a(x)
        x = self.conv_b(x)
		
        y = F.conv2d(y, abs(self.preconv_weight), abs(self.preconv_bias), 1, 2, (1,1), 1)
        y = self.conv_a_(y)
        y = self.conv_b_(y)
        
        y = self.coord(y)
        x = x * y + x
		
        x = self.conv_c(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)       
        return x

    def reset_parameters(self):
        for mod in self.modules():
            if isinstance(mod, SRM_conv2d) or \
                    isinstance(mod, nn.BatchNorm2d) or \
                    isinstance(mod, ConvBlock):
                mod.reset_parameters()
            elif isinstance(mod, nn.Linear):
                nn.init.normal(mod.weight, 0. ,0.01)
                # nn.init.xavier_normal(mod.weight)
                mod.bias.data.zero_()
                

class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def train(model, device, train_loader, optimizer, epoch):
  batch_time = AverageMeter() #ONE EPOCH TRAIN TIME
  data_time = AverageMeter()
  losses = AverageMeter()

  model.train()

  end = time.time()

  for i, sample in enumerate(train_loader):

    data_time.update(time.time() - end) 

    data, prob_data, label = sample['data'], sample['prob_data'], sample['label']

    shape = list(data.size())
    data = data.reshape(shape[0] * shape[1], *shape[2:])
    shape_prob = list(prob_data.size())
    prob_data = prob_data.reshape(shape_prob[0] * shape_prob[1], *shape_prob[2:])
    label = label.reshape(-1)


    data, prob_data, label = data.to(device), prob_data.to(device), label.to(device)

    optimizer.zero_grad()

    end = time.time()

    output = model(data, prob_data)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, label)

    losses.update(loss.item(), data.size(0))

    loss.backward()      #BP
    optimizer.step()

    batch_time.update(time.time() - end) #BATCH TIME = BATCH BP+FP
    end = time.time()

    if i % TRAIN_PRINT_FREQUENCY == 0:
      # logging.info('Epoch: [{}][{}/{}] \t Loss {:.6f}'.format(epoch, i, len(train_loader), loss.item()))

      logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

def adjust_bn_stats(model, device, train_loader):
  model.train()

  with torch.no_grad():
    for sample in train_loader:
      data, prob_data, label = sample['data'], sample['prob_data'], sample['label']

      shape = list(data.size())
      data = data.reshape(shape[0] * shape[1], *shape[2:])
      shape_prob = list(prob_data.size())
      prob_data = prob_data.reshape(shape_prob[0] * shape_prob[1], *shape_prob[2:])
      label = label.reshape(-1)

      data, prob_data, label = data.to(device), prob_data.to(device), label.to(device)

      output = model(data, prob_data)


def evaluate(model, device, eval_loader, epoch, optimizer, best_acc, PARAMS_PATH, PARAMS_PATH1, TMP):
  model.eval()

  test_loss = 0
  correct = 0

  with torch.no_grad():
    for sample in eval_loader:
      data, prob_data, label = sample['data'], sample['prob_data'], sample['label']

      shape = list(data.size())
      data = data.reshape(shape[0] * shape[1], *shape[2:])
      shape_prob = list(prob_data.size())
      prob_data = prob_data.reshape(shape_prob[0] * shape_prob[1], *shape_prob[2:])
      label = label.reshape(-1)

      data, prob_data, label = data.to(device), prob_data.to(device), label.to(device)

      output = model(data, prob_data)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(label.view_as(pred)).sum().item()

  accuracy = correct / (len(eval_loader.dataset) * 2)
  all_state = {
      'original_state': model.state_dict(),
      'optimizer_state': optimizer.state_dict(),
      'epoch': epoch
    }
  torch.save(all_state, PARAMS_PATH1)
  if accuracy > best_acc and epoch > TMP:
    best_acc = accuracy
    all_state = {
      'original_state': model.state_dict(),
      'optimizer_state': optimizer.state_dict(),
      'epoch': epoch
    }
    torch.save(all_state, PARAMS_PATH)
  
  logging.info('-' * 8)
  logging.info('Eval accuracy: {:.4f}'.format(accuracy))
  logging.info('Best accuracy:{:.4f}'.format(best_acc))   
  logging.info('-' * 8)
  return best_acc


#def initWeights(module):
  #if type(module) == nn.Conv2d:
    #if module.weight.requires_grad:
      #nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

  #if type(module) == nn.Linear:
    #nn.init.normal_(module.weight.data, mean=0, std=0.01)
    #nn.init.constant_(module.bias.data, val=0)
    

class AugData():
  def __call__(self, sample):
    data, prob_data, label = sample['data'], sample['prob_data'], sample['label']

    rot = random.randint(0,3)

    data = np.rot90(data, rot, axes=[1, 2]).copy()
    prob_data = np.rot90(prob_data, rot, axes=[1, 2]).copy()

    if random.random() < 0.5:
      data = np.flip(data, axis=2).copy()
      prob_data = np.flip(prob_data, axis=2).copy()

    new_sample = {'data': data, 'prob_data': prob_data, 'label': label}

    return new_sample


class ToTensor():
  def __call__(self, sample):
    data, prob_data, label = sample['data'], sample['prob_data'], sample['label']

    data = np.expand_dims(data, axis=1)
    data = data.astype(np.float32)
    prob_data = np.expand_dims(prob_data, axis=1)
    prob_data = prob_data.astype(np.float32)
    # data = data / 255.0

    new_sample = {
      'data': torch.from_numpy(data),
      'prob_data': torch.from_numpy(prob_data),
      'label': torch.from_numpy(label).long(),
    }

    return new_sample


class MyDataset(Dataset):
  def __init__(self, index_path, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, boss_cover_prob, boss_stego_prob, bows_cover_prob, bows_stego_prob, transform=None):
    self.index_list = np.load(index_path)
    self.transform = transform

    self.bossbase_cover_path = BOSSBASE_COVER_DIR + '/{}.mat'
    self.bossbase_stego_path = BOSSBASE_STEGO_DIR + '/{}.mat'

    self.bows_cover_path = BOWS_COVER_DIR + '/{}.mat'
    self.bows_stego_path = BOWS_STEGO_DIR + '/{}.mat'
    
    self.boss_cover_prob = boss_cover_prob + '/{}.mat'
    self.boss_stego_prob = boss_stego_prob + '/{}.mat'
        
    self.bows_cover_prob = bows_cover_prob + '/{}.mat'
    self.bows_stego_prob = bows_stego_prob + '/{}.mat'

  def __len__(self):
    return self.index_list.shape[0]

  def __getitem__(self, idx):
    file_index = self.index_list[idx]

    if file_index <= 10000:
      cover_path = self.bossbase_cover_path.format(file_index)
      stego_path = self.bossbase_stego_path.format(file_index)
      cover_prob_path = self.boss_cover_prob.format(file_index)
      stego_prob_path = self.boss_stego_prob.format(file_index)
    else:
      cover_path = self.bows_cover_path.format(file_index - 10000)
      stego_path = self.bows_stego_path.format(file_index - 10000)
      cover_prob_path = self.bows_cover_prob.format(file_index - 10000)
      stego_prob_path = self.bows_stego_prob.format(file_index - 10000)


    #cover_data = cv2.imread(cover_path, -1)
    #stego_data = cv2.imread(stego_path, -1)
    cover_data = sio.loadmat(cover_path)['img']
    stego_data = sio.loadmat(stego_path)['img']
    cover_prob_data = sio.loadmat(cover_prob_path)['img']
    stego_prob_data = sio.loadmat(stego_prob_path)['img']


    data = np.stack([cover_data, stego_data])
    prob_data = np.stack([cover_prob_data, stego_prob_data])
    label = np.array([0, 1], dtype='int32')

    sample = {'data': data, 'prob_data': prob_data, 'label': label}

    if self.transform:
      sample = self.transform(sample)

    return sample


def setLogger(log_path, mode='a'):
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  if not logger.handlers:
    # Logging to a file
    file_handler = logging.FileHandler(log_path, mode=mode)
    file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)


def main(args):

#  setLogger(LOG_PATH, mode='w')

#  Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
  statePath = args.statePath

  device = torch.device("cuda")

  kwargs = {'num_workers': 1, 'pin_memory': True}

  train_transform = transforms.Compose([
    AugData(),
    ToTensor()
  ])

  eval_transform = transforms.Compose([
    ToTensor()
  ])

  DATASET_INDEX = args.DATASET_INDEX
  STEGANOGRAPHY = args.STEGANOGRAPHY
  EMBEDDING_RATE = args.EMBEDDING_RATE
  JPEG_QUALITY = args.JPEG_QUALITY

  BOSSBASE_COVER_DIR = '/home/weikangkang/data/BossBase/BossBase-1.01-cover-resample-256-jpeg-{}-non-rounded'.format(JPEG_QUALITY)
  BOSSBASE_STEGO_DIR = '/home/weikangkang/data/BossBase/BossBase-1.01-{}-{}-resample-256-jpeg-{}-non-rounded'.format(STEGANOGRAPHY, EMBEDDING_RATE, JPEG_QUALITY)

  BOWS_COVER_DIR = '/home/weikangkang/data/BossBase/BOWS2-cover-resample-256-jpeg-{}-non-rounded'.format(JPEG_QUALITY)
  BOWS_STEGO_DIR = '/home/weikangkang/data/BossBase/BOWS2-{}-{}-resample-256-jpeg-{}-non-rounded'.format(STEGANOGRAPHY, EMBEDDING_RATE, JPEG_QUALITY)
  
  boss_cover_prob = '/data/wkk/BossBase_data/BossBase/BossBase-1.01-cover-{}-{}-resample-256-jpeg-{}_prob-non-rounded'.format(STEGANOGRAPHY, EMBEDDING_RATE, JPEG_QUALITY)
  boss_stego_prob = '/data/wkk/BossBase_data/BossBase/BossBase-1.01-stego-{}-{}-resample-256-jpeg-{}_prob-non-rounded'.format(STEGANOGRAPHY, EMBEDDING_RATE, JPEG_QUALITY)
    
  bows_cover_prob = '/data/wkk/BossBase_data/BossBase/BOWS2-cover-{}-{}-resample-256-jpeg-{}_prob-non-rounded'.format(STEGANOGRAPHY, EMBEDDING_RATE, JPEG_QUALITY)
  bows_stego_prob = '/data/wkk/BossBase_data/BossBase/BOWS2-stego-{}-{}-resample-256-jpeg-{}_prob-non-rounded'.format(STEGANOGRAPHY, EMBEDDING_RATE, JPEG_QUALITY)
  

  TRAIN_INDEX_PATH = 'index_list{}/bossbase_and_bows_train_index.npy'.format(DATASET_INDEX)
  VALID_INDEX_PATH = 'index_list{}/bossbase_valid_index.npy'.format(DATASET_INDEX)
  TEST_INDEX_PATH = 'index_list{}/bossbase_test_index.npy'.format(DATASET_INDEX)

  LOAD_RATE = float(EMBEDDING_RATE) + 0.1   
  LOAD_RATE = round(LOAD_RATE, 1)
  
  global LR
  global DECAY_EPOCH
  global EPOCHS
  global TMP

  if LOAD_RATE != 0.5 and JPEG_QUALITY=='75': 
    LR = 0.001
    DECAY_EPOCH = [50, 80]
    EPOCHS = 110
    TMP = 80

  if JPEG_QUALITY=='95': 
    LR = 0.001
    DECAY_EPOCH = [50, 80]
    EPOCHS = 110
    TMP = 80
    
  if LOAD_RATE == 0.5 and JPEG_QUALITY=='95': 
    LR = 0.001 
    DECAY_EPOCH = [130, 230]
    EPOCHS = 280
    TMP = 230

  PARAMS_NAME = '{}-{}-{}-params_{}.pt'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX,JPEG_QUALITY)
  LOG_NAME = '{}-{}-{}-model_log_{}'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX,JPEG_QUALITY)
  PARAMS_NAME1 = '{}-{}-{}-process-params_{}.pt'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX,JPEG_QUALITY)
  #statePath='./SCA_JYeNet_test2/'+PARAMS_NAME1
  
  PARAMS_PATH = os.path.join(OUTPUT_PATH, PARAMS_NAME)
  PARAMS_PATH1 = os.path.join(OUTPUT_PATH, PARAMS_NAME1)
  LOG_PATH = os.path.join(OUTPUT_PATH, LOG_NAME)

  #transfer learning 
  PARAMS_INIT_NAME = '{}-{}-{}-params_{}.pt'.format(STEGANOGRAPHY, LOAD_RATE, DATASET_INDEX, JPEG_QUALITY)
  
  if LOAD_RATE == 0.5 and JPEG_QUALITY == '95':
    PARAMS_INIT_NAME = '{}-{}-{}-params_{}.pt'.format(STEGANOGRAPHY, '0.4', DATASET_INDEX, '75')
    #PARAMS_INIT_NAME = '{}-{}-{}-params-{}-lr={}-{}-{}-{}_{}.pt'.format(STEGANOGRAPHY, LOAD_RATE, DATASET_INDEX, TIMES, '0.005', 80,140,180,JPEG_QUALITY)
   
  PARAMS_INIT_PATH = os.path.join(OUTPUT_PATH, PARAMS_INIT_NAME)
  print(PARAMS_INIT_PATH)

  setLogger(LOG_PATH, mode='w')

  Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
  
  train_dataset = MyDataset(TRAIN_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, boss_cover_prob, boss_stego_prob, bows_cover_prob, bows_stego_prob,train_transform)
  valid_dataset = MyDataset(VALID_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, boss_cover_prob, boss_stego_prob, bows_cover_prob, bows_stego_prob,eval_transform)
  test_dataset = MyDataset(TEST_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, boss_cover_prob, boss_stego_prob, bows_cover_prob, bows_stego_prob,eval_transform)

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
  valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)


  model = Net().to(device)  
  #model.apply(initWeights) 
  params = model.parameters()

  #hpf_params = list(map(id, model.group1.parameters()))
  #res_params = filter(lambda p: id(p) not in hpf_params, model.parameters())
       
  #param_groups = [{'params': res_params, 'weight_decay': WEIGHT_DECAY},
                    #{'params': model.group1.parameters()}]

  optimizer = optim.Adamax(params, lr=LR, weight_decay=WEIGHT_DECAY)
  #optimizer = optim.Adamax(param_groups, lr=LR)

  # optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=0.9)

  if statePath:
    logging.info('-' * 8)
    logging.info('Load state_dict in {}'.format(statePath))
    logging.info('-' * 8)

    all_state = torch.load(statePath)

    original_state = all_state['original_state']
    optimizer_state = all_state['optimizer_state']
    epoch = all_state['epoch']

    model.load_state_dict(original_state)
    optimizer.load_state_dict(optimizer_state)

    startEpoch = epoch + 1

  else:
    startEpoch = 1
  
  if LOAD_RATE != 0.5:
    all_state = torch.load(PARAMS_INIT_PATH)
    original_state = all_state['original_state']
    model.load_state_dict(original_state)
  
  if LOAD_RATE == 0.5 and JPEG_QUALITY=='95':
    all_state = torch.load(PARAMS_INIT_PATH)
    original_state = all_state['original_state']
    model.load_state_dict(original_state)
      
  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
  best_acc = 0.0
  for epoch in range(startEpoch, EPOCHS + 1):
    scheduler.step()

    train(model, device, train_loader, optimizer, epoch)

    if epoch % EVAL_PRINT_FREQUENCY == 0:
      adjust_bn_stats(model, device, train_loader)
      best_acc = evaluate(model, device, valid_loader, epoch, optimizer, best_acc, PARAMS_PATH, PARAMS_PATH1, TMP)

  logging.info('\nTest set accuracy: \n')

   #load best parmater to test    
  all_state = torch.load(PARAMS_PATH)
  original_state = all_state['original_state']
  optimizer_state = all_state['optimizer_state']
  model.load_state_dict(original_state)
  optimizer.load_state_dict(optimizer_state)

  adjust_bn_stats(model, device, train_loader)
  evaluate(model, device, test_loader, epoch, optimizer, best_acc, PARAMS_PATH, PARAMS_PATH1, TMP)


def myParseArgs():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '-i',
    '--DATASET_INDEX',
    help='Path for loading dataset',
    type=str,
    default='1'
  )

  parser.add_argument(
    '-alg',
    '--STEGANOGRAPHY',
    help='embedding_algorithm',
    type=str,
    choices=['j-uniward','UED'],
    required=True
  )

  parser.add_argument(
    '-rate',
    '--EMBEDDING_RATE',
    help='embedding_rate',
    type=str,
    choices=['0.2', '0.3', '0.4'],
    required=True
  )

  parser.add_argument(
    '-quality',
    '--JPEG_QUALITY',
    help='JPEG_QUALITY',
    type=str,
    choices=['75', '95'],
    required=True
  )

  parser.add_argument(
    '-g',
    '--gpuNum',
    help='Determine which gpu to use',
    type=str,
    choices=['0', '1', '2', '3'],
    required=True
  )

  parser.add_argument(
    '-l',
    '--statePath',
    help='Path for loading model state',
    type=str,
    default=''
  )

  args = parser.parse_args()

  return args


if __name__ == '__main__':
  args = myParseArgs()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuNum
  main(args)
