from __future__ import print_function, division
import os,time
from warpctc_pytorch import CTCLoss
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 定义 LSTM 层 
import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms
from torchsample.callbacks import EarlyStopping

# 工具类
def readfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split()
        dic[p[0]] = p[1:]
    return dic
# 定义数据加载代码
#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys,glob
from PIL import Image
import numpy as np


class TxtDataset(Dataset):

    def __init__(self,image_root="",maxlabellength=8, txtfile=None, transform=None, target_transform=None):
        # 读取所有行
        self.sample_dict = readfile(txtfile)        
        self.nSamples = len(self.sample_dict)
        self.images = list(self.sample_dict.keys())
        if image_root != "":
            self.images = [image_root + imgfile for imgfile in self.images]
        self.labels = list(self.sample_dict.values())
        self.transform = transform
        self.target_transform = target_transform
        self.maxlabellen = maxlabellength

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        
        img = self.images[index]
        label = self.labels[index]
        try:
            img = Image.open(img).convert('L')
        except IOError:
            print('Corrupted image for %d' % index)
            pass

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
        
        length = len(label)
        
#         if length < self.maxlabellen:
#             for i in range(self.maxlabellen - length):
#                 label.append("0")                
        
        label = torch.Tensor(list(map(int,label))).int()
        return (img, label,length,self.images[index])


# # 测试
# test_ds = TxtDataset(image_root="/home/creatorai/workspace/1016_detect/crnn.pytorch/data/labelimageTrans/",txtfile=('../chinese_ocr/train/test_hp_new_0102.txt'),
#                                     transform=transforms.Compose([
#             #transforms.CenterCrop((128,256)),
#             transforms.Resize((32,280)),
#             #transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#                                     target_transform=None)
# ds = torch.utils.data.DataLoader(test_ds,batch_size=4,shuffle=False)
# for data in ds:
#     print(data)
  
    
# 定义 Densenet 网络
# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, growth_rate=8, block_config=(8, 8, 8), compression=0.5,
                 num_init_features=64, bn_size=4, drop_rate=0.2,
                 num_classes=10, small_inputs=False, efficient=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(1, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(1, num_init_features, kernel_size=5, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
#            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
#                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            
            self.features.add_module('denseblock%d' % (i + 1), block)
            
            num_features = num_features + num_layers * growth_rate
            
            
            
            if i != len(block_config) - 1:
                
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)
            
                

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        #print(out.shape)
        #out = F.avg_pool2d(out, kernel_size=(7,1))#.view(features.size(0), -1)
        #print(out.shape)
        #out = self.classifier(out)
        return out
    


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output
# 定义 CRNN 
class CRNN1(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN1, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output
class CRNN(nn.Module):

    def __init__(self, growth_rate=8, block_config=(8, 8, 8), compression=0.5,
                 num_init_features=64, bn_size=4, drop_rate=0.2,
                 num_classes=10, nh=256, small_inputs=False, efficient=False):
        super(CRNN, self).__init__()
        #assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.cnn = DenseNet(growth_rate=8, block_config=(8, 8, 8), compression=0.5,
                 num_init_features=64, bn_size=4, drop_rate=0.2, small_inputs=False, efficient=False)
        self.dense = nn.Linear(1024, num_classes)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(1024, nh, nh),
            BidirectionalLSTM(nh, nh, num_classes))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        #print(b,c,h,w)
        conv = conv.view((b,-1,26))
        conv = conv.permute(2, 0, 1) # [w, b, c]
        w, b, c = conv.size()
        #output = self.dense(conv)
#         print(b,c,h,w)
#         assert h == 1, "the height of conv must be 1"
        #conv = conv.squeeze(2)
        #conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output
def decode(text, length, raw=False):
    if length.numel() == 1:
        length = length[0]
        
        assert text.numel() == length, "text with length: {} does not match declared length: {}".format(text.numel(), length)
        if raw:
            return ''.join([alphabet[i - 1] for i in text]),[i-1 for i in text]
        else:
            char_list = []
            ind_list = []
            for i in range(length):
                if text[i] != 0 and (not (i > 0 and text[i - 1] == text[i])):
                    char_list.append(alphabet[text[i] - 1])
                    ind_list.append(text[i] - 1)
            return (''.join(char_list),ind_list)
    else:
        # batch mode
        assert text.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(text.numel(), length.sum())
        texts = []
        index = 0
        for i in range(length.numel()):
            l = length[i]
            texts.append(
                decode(
                    text[index:index + l], torch.IntTensor([l]), raw=raw))
            index += l
        return texts

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
# 定义 train 代码
def train(model, criterion, optimizer, scheduler, num_epochs, use_gpu):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    loss_mean = []
    for epoch in range(num_epochs):
        begin_time = time.time()
        count_batch = 0
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            count_batch = 0
            if phase == 'train':
                #scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloders[phase]:
                count_batch += 1
                # get the inputs
                inputs, labels,lengths,imgfile = data
                #print(count_batch)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels)
                    lengths = Variable(lengths.int())
                else:
                    inputs, labels ,lengths  = Variable(inputs), Variable(labels), Variable(lengths)
                # zero the parameter gradients
                
                batch_size_tp = inputs.size(0)
                # forward
                outputs = model(inputs)
                preds_size = Variable(torch.IntTensor([outputs.size(0)] * batch_size_tp))
                #print((outputs),(preds_size),(lengths),(labels.view(-1)))
                loss = criterion(outputs, labels.view(-1), preds_size, lengths)  / batch_size_tp

                #print(loss.data[0])
                #print(""+9)
#                 outputs = torch.squeeze(outputs)
#                 vc = torch.squeeze(vc)
#                 _, preds = torch.max(outputs.data, 1)
#                 loss = criterion(outputs,vc, labels ,split_y)

                # backward + optimize only if in training phase
                if phase == 'train':
                    model.zero_grad()
                    optimizer.zero_grad()
                    loss.backward()
                    #nn.utils.clip_grad_norm(model.parameters(), 10.0)
                    loss_mean.append(loss.data[0])
                    optimizer.step()
                # statistics
                #running_loss += loss.data[0]
                _, preds = torch.max(outputs.data, 2)
                preds = preds.view(-1)
#                 pred_list = decode(preds.data, preds_size.data, raw=False)
#                 label_list = decode(labels.view(-1).data, lengths.data, raw=False)
#                 for pred, target in zip(pred_list, label_list):
#                     if pred == target:
#                         running_corrects += 1
                #running_corrects += torch.sum(pred_list == label_list)
                running_corrects = 0
                # print result every 10 batch
                if count_batch%10 == 0:
                    running_loss = np.mean(loss_mean)
                    batch_loss = running_loss#running_loss / (batch_size*count_batch)
                    batch_acc = running_corrects / (batch_size*count_batch)
                    print('{} Epoch [{}] Batch [{}] Loss: {:.4f} Acc: {:.4f} Time: {:.4f}s'. \
                          format(phase, epoch, count_batch, batch_loss, batch_acc, time.time()-begin_time))
                    begin_time = time.time()
                    
            running_loss = np.mean(loss_mean)
            
            epoch_loss = running_loss#running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # save model
            if phase == 'train':
                if not os.path.exists('output'):
                    os.makedirs('output')
                torch.save(model, 'output/resnet_epoch{}.pkl'.format(epoch))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
            
            
            scheduler.step(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

class resizeNormalize(object):

    def __init__(self, size, aug=None,interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.aug = aug
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        if self.aug != None:
            img = self.aug(img)
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        #img.div_(255.0).add_(-0.5)
        img.sub_(0.5).div_(0.5)  # / 255.0 - 0.5
        return img

def text_collate(batch):
    img = list()
    seq = list()
    seq_len = list()
    files = list()
    for sample in batch: #(img, label,length,self.images[index])
        img.append(sample[0].float())
        seq.extend(sample[1])
        seq_len.append(sample[2])
        files.append(sample[3])
    img = torch.stack(img)
    seq = torch.Tensor(seq).int()
    seq_len = torch.Tensor(seq_len).int()
    return (img,seq,seq_len,files)
 
import Augmentor
from Augmentor.Operations import *

import cv2 as cv
import numpy as np
import PIL

# Create your new operation by inheriting from the Operation superclass:

#整体模糊图片类
class FoldImage(Operation):
    # Here you can accept as many custom parameters as required:
    def __init__(self, probability, degree=7, angle=20):
        # degree 调整范围是1到7的int类型，左右闭区间（基于此文件夹图片效果）
        # Call the superclass's constructor (meaning you must
        # supply a probability value):
        Operation.__init__(self, probability)
        # Set your custom operation's member variables here as required:
        self.degree = degree
        self.angle = angle

    # Your class must implement the perform_operation method:
    def perform_operation(self, image):
        # Start of code to perform custom image operation.
        #输入是cv.imread获得的图像
        #degree就是重影的程度，数字越大越严重
        #angle是重影的方向，从0到135逆时针旋转，0是右下45度，45是往右，90是左上45度，135是竖直往上
        #输出是uint8的np数组
        #image = np.array(image)
        target_list = []
        for img in image:
            #image_array = np.array(image).astype('uint8')
            image = np.asarray(img) #cv.cvtColor(np.asarray(img),cv.COLOR_RGB2BGR)

            # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
            M = cv.getRotationMatrix2D((self.degree / 2, self.degree / 2), self.angle, 1)
            motion_blur_kernel = np.diag(np.ones(self.degree))
            motion_blur_kernel = cv.warpAffine(motion_blur_kernel, M, (self.degree, self.degree))

            motion_blur_kernel = motion_blur_kernel / self.degree
            blurred = cv.filter2D(image, -1, motion_blur_kernel)

            # convert to uint8
            cv.normalize(blurred, blurred, 0, 255, cv.NORM_MINMAX)
            blurred = np.array(blurred, dtype=np.uint8)
            # End of code to perform custom image operation.
            image = PIL.Image.fromarray(blurred)
            target_list.append(image)
        # Return the image so that it can further processed in the pipeline:
        return target_list
    
#产生重影类    
class BlurImage(Operation):
    # Here you can accept as many custom parameters as required:
    def __init__(self, probability, degree=7):
        # degree 范围是1到7的int类型数，左右闭区间（依据此文件夹图片的效果）
        # Call the superclass's constructor (meaning you must
        # supply a probability value):
        Operation.__init__(self, probability)
        # Set your custom operation's member variables here as required:
        self.degree = degree


    # Your class must implement the perform_operation method:
    def perform_operation(self, image):
        # Start of code to perform custom image operation.
        #image_array = np.array(image).astype('uint8')
        #输入参数是cv.imread获得的图像
        #degree模糊的程度，越大越模糊
        #输出是uint8的array数组
        #image = np.array(image)
        target_list = []
        for img in image:
            #img = cv.imread(img)
            img = np.asarray(img)
            dst = cv.blur(img, (1, self.degree)) 
            dst = cv.medianBlur(dst,1)
            image = PIL.Image.fromarray(dst)
            target_list.append(image)
        # Return the image so that it can further processed in the pipeline:
        return target_list
    
    
    
#部分遮掩产生马赛克类
class CorrodeImage(Operation):
    # Here you can accept as many custom parameters as required:
    def __init__(self, probability, area_h=15, area_w=15):
        # degree 范围是1到7的int类型数，左右闭区间（依据此文件夹图片的效果）
        # Call the superclass's constructor (meaning you must
        # supply a probability value):
        Operation.__init__(self, probability)
        # Set your custom operation's member variables here as required:
        self.area_h = area_h
        self.area_w = area_w


    # Your class must implement the perform_operation method:
    def perform_operation(self, image):
        # Start of code to perform custom image operation.
        #image_array = np.array(image).astype('uint8')
        #输入参数是cv.imread获得的图像
        #degree模糊的程度，越大越模糊
        #输出是uint8的array数组
        #image = np.array(image)
        target_list = []
        for img in image:
            area_h = self.area_h
            area_w = self.area_w
            img = np.asarray(img)
            h,w = img.shape[:2]
            point_h = random.randint(1,h-area_h)
            point_w = random.randint(1,w-area_w)
            img.flags.writeable = True
            #print(point_h,point_w)
            roi = img[point_h:point_h+area_h,point_w:point_w+area_w]
            kernel = np.ones((5, 5), np.uint8)
            dst = cv.erode(roi, kernel)
            img[point_h:point_h+area_h,point_w:point_w+area_w] = dst
            size = 2
            for i in range(size):
                for j in range(size):
                    img[point_h+i][point_w+j]=img[point_h][point_w]
            image = PIL.Image.fromarray(img)
            target_list.append(image)
        # Return the image so that it can further processed in the pipeline:
        return target_list
    
if __name__ == '__main__':
    alphabet = u""" 京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学港澳使领警挂0123456789ABCDEFGHJKLMNOPQRSTUVWXYZ"""
    # 设置增强方式
    p = Augmentor.Pipeline()
    
    # 重影、模糊、马赛克增强
#     fold = FoldImage(probability=0.5)#min_factor=0, max_factor=1)
#     p.add_operation(fold)
#     blur = BlurImage(probability=0.5)#min_factor=0, max_factor=1)
#     p.add_operation(blur)
#     corr = CorrodeImage(probability=0.5)#min_factor=0, max_factor=1)
#     p.add_operation(corr)
    # 重影、模糊、马赛克增强
    
    eras = RandomErasing(probability=0.7, rectangle_area=0.1)#min_factor=0, max_factor=1)
    p.add_operation(eras)
    bright = RandomBrightness(probability=0.4, min_factor=0.4, max_factor=1.5) #(probability=0.7, rectangle_area=0.1)#min_factor=0, max_factor=1)
    p.add_operation(bright)
    color = RandomColor(probability=0.4, min_factor=0.5, max_factor=0.8) #(probability=0.7, rectangle_area=0.1)#min_factor=0, max_factor=1)
    p.add_operation(color)
    contrast = RandomContrast(probability=0.4, min_factor=0.5, max_factor=0.9) #(probability=0.7, rectangle_area=0.1)#min_factor=0, max_factor=1)
    p.add_operation(contrast)
    #p.rotate(probability=0.7, max_left_rotation=5, max_right_rotation=5) #_without_crop , expand=False
    #p.zoom(probability=0.5, min_factor=0.9, max_factor=1.1)
    #p.skew_tilt(probability=1.0,magnitude=0.1)
    #p.shear(probability=1.0,max_shear_left=5, max_shear_right=5)
    
    data_transforms = {
        'train': transforms.Compose([
            #transforms.CenterCrop((128,256)),
            transforms.Resize((32,280)),
            p.torch_transform(),
            #ransforms.Scale(256),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((32,280)),
            #transforms.Scale(256),
            #transforms.CenterCrop((256,244)),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    

    use_gpu = torch.cuda.is_available()

    batch_size = 192
    num_class = len(alphabet)

    image_datasets_train = TxtDataset(image_root="",txtfile=('./train/train_hp_fake_bo14_0117.txt'),
                                      transform=resizeNormalize((208, 64),p.torch_transform()), #,p.torch_transform()
                                      target_transform=None)

    train_ds = torch.utils.data.DataLoader(image_datasets_train,batch_size=batch_size,shuffle=True,collate_fn=text_collate)

    image_datasets_test = TxtDataset(image_root="",txtfile=('./train/test_hp_fake_bo14_0117.txt'),
                                      transform=resizeNormalize((208, 64)),
                                      target_transform=None)

    test_ds = torch.utils.data.DataLoader(image_datasets_test,batch_size=16,shuffle=False,collate_fn=text_collate)

    dataloders = {'train': train_ds, 'val': test_ds}
    dataset_sizes = {'train': len(image_datasets_train), 'val': len(image_datasets_test)}

    # get model and replace the original fc layer with your fc layer
#     model_ft = CRNN(growth_rate=8, block_config=(8, 8, 8), compression=0.5,
#                  num_init_features=64, bn_size=4, drop_rate=0.2,
#                  num_classes=num_class, nh=256, small_inputs=False, efficient=False)
    
    model_ft = torch.load('output/best_densenet_plate_rec_fake_bo17_26width_kf60w_L_01.pkl') # best_densenet_plate_rec_h64_02_aug_02
    #model_ft = CRNN1(32, 1, nclass=num_class,nh= 256)
    #model_ft.apply(weights_init)
    # define cost function
    criterion = CTCLoss()
    # if use gpu
    if use_gpu:
        model_ft = model_ft.cuda()
        criterion = criterion.cuda()

    
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)
    #optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001 , betas=(0.5, 0.999)) # 初始0.01
    # Decay LR by a factor of 0.2 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.2)
    #exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, 'min',factor=0.5, patience=1, threshold=0.01, threshold_mode='rel', cooldown=0, min_lr=0.0000001,verbose=True)
    

    #multi-GPU
    #model_ft = torch.nn.DataParallel(model_ft, device_ids=[0])

    #train model
    model_ft = train(model=model_ft,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=10,
                           use_gpu=use_gpu)

    #save best model
    torch.save(model_ft,"output/best_densenet_plate_rec_fake_bo17_26width_bo14_L_01.pkl")
