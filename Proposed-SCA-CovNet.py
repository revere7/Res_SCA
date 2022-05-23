#!/usr/bin/env python3

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
import copy
import logging
import random
import scipy.io as sio
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from CovNet_SRM_filter import all_normalized_hpf_list
from MPNCOV.python import MPNCOV

IMAGE_SIZE = 256
BATCH_SIZE = 32 // 2
EPOCHS = 200
LR = 0.01
WEIGHT_DECAY = 5e-4

TRAIN_FILE_COUNT = 14000
TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1
DECAY_EPOCH = [80, 140, 180]

OUTPUT_PATH = Path(__file__).stem


class TLU(nn.Module):
  def __init__(self, threshold):
    super(TLU, self).__init__()

    self.threshold = threshold

  def forward(self, input):
    output = torch.clamp(input, min=-self.threshold, max=self.threshold)

    return output
    
    
class HPF(nn.Module):
  def __init__(self):
    super(HPF, self).__init__()

    all_hpf_list_5x5 = []

    for hpf_item in all_normalized_hpf_list:
      if hpf_item.shape[0] == 3:
        hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

      all_hpf_list_5x5.append(hpf_item)
      
    #all_hpf_list_5x5 = all_hpf_list_5x5
    hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)

    self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
    self.hpf.weight = hpf_weight

    self.tlu = TLU(3.0)


  def forward(self, input):

    output = self.hpf(input)
    output = self.tlu(output)
    
    #output_sca = torch.abs(self.hpf(prob_data))
    #output = output + output_sca
    
    return output
    
    
class HPF_sca(nn.Module):
  def __init__(self):
    super(HPF_sca, self).__init__()

    all_hpf_list_5x5 = []

    for hpf_item in all_normalized_hpf_list:
      if hpf_item.shape[0] == 3:
        hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

      all_hpf_list_5x5.append(hpf_item)
      
    #all_hpf_list_5x5 = all_hpf_list_5x5
    hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)

    self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
    self.hpf.weight = hpf_weight


  def forward(self, prob_data):
    
    output_sca = self.hpf(prob_data)
    output = output_sca
    
    return output
    
    
class HPF_RGSA(nn.Module):
  def __init__(self):
    super(HPF_RGSA, self).__init__()

    all_hpf_list_5x5 = []

    for hpf_item in all_normalized_hpf_list:
      if hpf_item.shape[0] == 3:
        hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

      all_hpf_list_5x5.append(hpf_item)
      
    #all_hpf_list_5x5 = all_hpf_list_5x5
    hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)

    self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
    self.hpf.weight = hpf_weight


  def forward(self, input):

    output = self.hpf(input)
    
    return output


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
    def __init__(self, inp, oup, reduction=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        #mip = max(8, inp // reduction)
        mip = max(16, inp // reduction)

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
                
      
class RGSA(nn.Module):
    def __init__(self, in_channels):
        super(RGSA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), 
            nn.Conv2d(32, 32, 3, padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.pool=nn.AvgPool2d(kernel_size=3, padding=1, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_srm):
        fea1 = self.conv1(x_srm)        
        fea2 = self.conv2(fea1)
        fea2 = self.pool(fea2)
        
        return fea2
        
        
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.group1 = HPF()
    self.group11 = HPF_sca()
    self.rgsa = RGSA(30)

    self.coord = CoordAtt(32, 32, reduction=4)

    self.group2 = nn.Sequential(
      nn.Conv2d(30, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),

      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),

      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),

      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),

      nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
      
    )

    self.group3 = nn.Sequential(
      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),

      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),

      nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
    )

    self.group4 = nn.Sequential(
      nn.Conv2d(64, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),

      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),

      nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
    )

    self.group5 = nn.Sequential(
      nn.Conv2d(128, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),

      nn.Conv2d(128, 256, kernel_size=3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),
    )

    self.fc1 = nn.Linear(int(256 * (256 + 1) / 2), 2)

  def forward(self, input, prob_data):
    output = input

    output_main = self.group1(output)
    output_sca = self.group11(prob_data)
    
    output_rgsa = self.rgsa(output_sca)
    out = self.coord(output_rgsa)
    
    output = self.group2(output_main)    
    output = output * out + output
    
    output = self.group3(output)
    output = self.group4(output)
    output = self.group5(output)


    output = MPNCOV.CovpoolLayer(output)
    output = MPNCOV.SqrtmLayer(output, 5)
    output = MPNCOV.TriuvecLayer(output)

    output = output.view(output.size(0), -1)
    output = self.fc1(output)

    return output


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
    batch_time = AverageMeter()
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
        output = model(data, prob_data)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)

        losses.update(loss.item(), data.size(0))

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
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


def evaluate(model, device, eval_loader, epoch, optimizer, best_acc, PARAMS_PATH,PARAMS_PATH1):
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
  

    if accuracy > best_acc and epoch > 180:
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



def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

            # nn.init.xavier_uniform_(module.weight.data)
            # nn.init.constant_(module.bias.data, val=0.2)
        # else:
        #   module.weight.requires_grad = True

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)
        nn.init.constant_(module.bias.data, val=0)

    if type(module) == nn.BatchNorm2d:
        nn.init.constant_(module.weight.data, val=1)
        nn.init.constant_(module.bias.data, val=0)


class AugData():
    def __call__(self, sample):
        data, prob_data, label = sample['data'], sample['prob_data'], sample['label']

        rot = random.randint(0, 3)

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
    def __init__(self, index_path, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, Alaska_cover_prob, Alaska_stego_prob, bows_cover_prob, bows_stego_prob,  transform=None):
        self.index_list = np.load(index_path)
        self.transform = transform
        self.bossbase_cover_path = BOSSBASE_COVER_DIR + '/{}.pgm'
        self.bossbase_stego_path = BOSSBASE_STEGO_DIR + '/{}.pgm'
        
        self.bows_cover_path = BOWS_COVER_DIR + '/{}.pgm'
        self.bows_stego_path = BOWS_STEGO_DIR + '/{}.pgm'

        self.alaska_cover_prob = Alaska_cover_prob + '/{}.mat'
        self.alaska_stego_prob = Alaska_stego_prob + '/{}.mat'
        
        self.bows_cover_prob = bows_cover_prob + '/{}.mat'
        self.bows_stego_prob = bows_stego_prob + '/{}.mat'

    def __len__(self):
        return self.index_list.shape[0]

    def __getitem__(self, idx):
        file_index = self.index_list[idx]
        
        if file_index <= 10000:
          cover_path = self.bossbase_cover_path.format(file_index)
          stego_path = self.bossbase_stego_path.format(file_index)
          cover_prob_path = self.alaska_cover_prob.format(file_index)
          stego_prob_path = self.alaska_stego_prob.format(file_index)
        else:
          cover_path = self.bows_cover_path.format(file_index - 10000)
          stego_path = self.bows_stego_path.format(file_index - 10000)
          cover_prob_path = self.bows_cover_prob.format(file_index - 10000)
          stego_prob_path = self.bows_stego_prob.format(file_index - 10000)

        cover_data = cv2.imread(cover_path, -1) 
        #cover_data = np.transpose(cover_data, (2, 0, 1)) 
        stego_data = cv2.imread(stego_path, -1)
        #stego_data = np.transpose(stego_data, (2, 0, 1))
        cover_prob_data = sio.loadmat(cover_prob_path)['pro']
        stego_prob_data = sio.loadmat(stego_prob_path)['pro']

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
    # setLogger(LOG_PATH, mode='w')

    # Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
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

    TRAIN_INDEX_PATH = 'index_list{}/bossbase_and_bows_train_index.npy'.format(DATASET_INDEX)
    VALID_INDEX_PATH = 'index_list{}/bossbase_valid_index.npy'.format(DATASET_INDEX)
    TEST_INDEX_PATH = 'index_list{}/bossbase_test_index.npy'.format(DATASET_INDEX)

    BOSSBASE_COVER_DIR = '/data/wkk/BossBose_data/BossBase-1.01-cover-resample-256'
    BOSSBASE_STEGO_DIR = '/data/wkk/BossBose_data/BossBase-1.01-{}-{}-resample-256'.format(STEGANOGRAPHY, EMBEDDING_RATE)
    
    BOWS_COVER_DIR = '/data/wkk/BossBose_data/BOWS2-cover-resample-256'
    BOWS_STEGO_DIR = '/data/wkk/BossBose_data/BOWS2-{}-{}-resample-256'.format(STEGANOGRAPHY, EMBEDDING_RATE)

    Alaska_cover_prob = '/data/wkk/BossBose_data/BossBase-1.01-cover-{}-{}-resample-256-prob'.format(STEGANOGRAPHY, EMBEDDING_RATE)
    Alaska_stego_prob = '/data/wkk/BossBose_data/BossBase-1.01-stego-{}-{}-resample-256-prob'.format(STEGANOGRAPHY, EMBEDDING_RATE)
    
    bows_cover_prob = '/data/wkk/BossBose_data/BOWS2-cover-{}-{}-resample-256-prob'.format(STEGANOGRAPHY, EMBEDDING_RATE)
    bows_stego_prob = '/data/wkk/BossBose_data/BOWS2-stego-{}-{}-resample-256-prob'.format(STEGANOGRAPHY, EMBEDDING_RATE)

    train_dataset = MyDataset(TRAIN_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, Alaska_cover_prob, Alaska_stego_prob, bows_cover_prob, bows_stego_prob, train_transform)
    valid_dataset = MyDataset(VALID_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, Alaska_cover_prob, Alaska_stego_prob,  bows_cover_prob, bows_stego_prob, eval_transform)
    test_dataset = MyDataset(TEST_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, Alaska_cover_prob, Alaska_stego_prob,  bows_cover_prob, bows_stego_prob, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    PARAMS_NAME = '{}-{}-{}-params.pt'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX)
    LOG_NAME = '{}-{}-{}-model_log'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX)
    PARAMS_NAME1 = '{}-{}-{}-process-params.pt'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX)
  
    PARAMS_PATH = os.path.join(OUTPUT_PATH, PARAMS_NAME)
    PARAMS_PATH1 = os.path.join(OUTPUT_PATH, PARAMS_NAME1)
    LOG_PATH = os.path.join(OUTPUT_PATH, LOG_NAME)

    setLogger(LOG_PATH, mode='w')

    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    model = Net().to(device)
    model.apply(initWeights)


    params = model.parameters()


    params_wd, params_rest = [], []
    for param_item in params:
        if param_item.requires_grad:
            (params_wd if param_item.dim() != 1 else params_rest).append(param_item)

    param_groups = [{'params': params_wd, 'weight_decay': WEIGHT_DECAY},
                      {'params': params_rest}]

    optimizer = optim.SGD(param_groups, lr=LR, momentum=0.9)

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

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)

    best_acc = 0.0
    for epoch in range(startEpoch, EPOCHS + 1):
      scheduler.step()

      train(model, device, train_loader, optimizer, epoch)

      if epoch % EVAL_PRINT_FREQUENCY == 0:
        adjust_bn_stats(model, device, train_loader)
        best_acc = evaluate(model, device, valid_loader, epoch, optimizer, best_acc, PARAMS_PATH,PARAMS_PATH1)

    logging.info('\nTest set accuracy: \n')

   #load best parmater to test    
    all_state = torch.load(PARAMS_PATH)
    original_state = all_state['original_state']
    optimizer_state = all_state['optimizer_state']
    model.load_state_dict(original_state)
    optimizer.load_state_dict(optimizer_state)

    adjust_bn_stats(model, device, train_loader)
    evaluate(model, device, test_loader, epoch, optimizer, best_acc, PARAMS_PATH,PARAMS_PATH1)


def myParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument(
      '-i',
      '--DATASET_INDEX',
      help='Path for loading dataset',
      type=str,
      default=''
    )

    parser.add_argument(
      '-alg',
      '--STEGANOGRAPHY',
      help='embedding_algorithm',
      type=str,
      choices=['HILL-CMDC', 's-uniward'],
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

