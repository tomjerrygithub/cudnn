from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms
import torch.nn.functional as F
import time
import os,cv2
from torch.utils.data import Dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from PIL import Image

# Some imports first
import torch.nn as nn
import math
import numpy as np
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd.variable import Variable
from torchvision import datasets, models, transforms

# 定义一个类，需要创建模型的时候，就实例化一个对象

class JamieSplitPlateModel(nn.Module):
    def __init__(self,Load_VIS_URL=None):
        super(JamieSplitPlateModel,self).__init__()
        
        # 开始，先交换出最后一层
        model_ft = models.resnet152(pretrained=False)
    
        num_final_in = model_ft.fc.in_features
        
        # edit 0109   1 channel
       # model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        model_ft.fc = nn.Conv2d(num_final_in, 512, 1)
        model_ft.avgpool = nn.AvgPool2d(kernel_size=3, stride=3, padding=0)              
        
        # add dropout
        model_ft.dp = nn.Dropout(p=0.3)
        
        # 现在，让我们定义新的一层，添加到最顶端       
        self.backbone = nn.Sequential(*list(model_ft.children()))
             
        #self.vertical_coordinate1 = nn.Conv2d(512, 32, 1)
        self.vertical_coordinate1 = nn.Linear(in_features=512 , out_features=32, bias=True)
        #self.vertical_coordinate2 = nn.Conv2d(512, 32, 1)
        self.vertical_coordinate2 = nn.Linear(in_features=512 , out_features=32, bias=True)
        self.score = nn.Conv2d(512, 2, 1)
        
    
    
    def forward(self,x):
        x = self.backbone(x)
        flatten = x.view(x.size(0), -1)
        vertical_coordinate1 = self.vertical_coordinate1(flatten)
        vertical_coordinate2 = self.vertical_coordinate2(flatten)
        score = self.score(x)

        return score,vertical_coordinate1,vertical_coordinate2

jmodel = JamieSplitPlateModel()
jmodel.train()


class JamieSplitPlateLoss(torch.nn.Module):
    
    def __init__(self):
        super(JamieSplitPlateLoss,self).__init__()
        self.Ls_cls = nn.CrossEntropyLoss()
        self.Ls_cls_v1 = nn.CrossEntropyLoss()
        self.Ls_cls_v2 = nn.CrossEntropyLoss()
        self.Lv_reg = nn.MSELoss() #.SmoothL1Loss() #
    
    def forward(self,score, vertical_pred1,vertical_pred2,label,split_y1,split_y2):
        clsloss = self.Ls_cls(score,label)
        cls_y1 = self.Ls_cls(vertical_pred1,split_y1)
        cls_y2 = self.Ls_cls(vertical_pred2,split_y2)
        #print(split_y1,split_y2)
        print("clsloss: ",clsloss.item(),"cls_y1: ",cls_y1.item(),"cls_y2: ",cls_y2.item())
        totloss = 0.1*clsloss  + 2.0*cls_y1 + 2.0*cls_y2##*0.1
        return totloss

# use PIL Image to read image
def default_loader(path):
#     try:
#         img = Image.open(path)
#         return img.convert('1')
#     except:
#         print("Cannot read image: {}".format(path))
    try:
        img = cv2.imread(path)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img = cv2.merge([img,img,img])
        img = Image.fromarray(img)
        #img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        return img
    except:
        print("Cannot read image: {}".format(path))    

def filter_cls(num):
    nums = num
    if nums > 31:
        nums = 31
    if nums < 0:
        nums = 0
    return nums
        
# define your Dataset. Assume each line in your .txt file is [name/tab/label], for example:0001.jpg 1
class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset = '', data_transforms=None, loader = default_loader):
        print(txt_path)
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            
            self.img_name = [ line.strip().split("*")[0].replace("./home","/data1/yukuifeng/home").replace("jamie/smalldataset1","creatorai/workspace")
                             .replace("/home/creatorai/workspace/labelyou/flask/examples/tutorial/flaskr/static/onlyone/","/home/creatorai/workspace/1016_detect/TextCNN/aug_code_data/4W_single_wrong/")
                             .replace("/home/creatorai/workspace/labelyou/flask/examples/tutorial/flaskr/static/double/","/home/creatorai/workspace/1016_detect/TextCNN/aug_code_data/4W_double_wrong/")
                             .replace("/data/xiesibo/exchange/yukuifeng/aug_image_0104_8plus/","/home/creatorai/workspace/1016_detect/TextCNN/aug_code_data/aug_image_0104_8plus/")
                             .replace("/data/xiesibo/exchange/yukuifeng/aug_image_0108_10plus/","/home/creatorai/workspace/1016_detect/TextCNN/aug_code_data/aug_image_0108_10plus/")
                             .replace("/data/xiesibo/exchange/yukuifeng/police_0108_70plus/","/home/creatorai/workspace/1016_detect/TextCNN/aug_code_data/police_0108_70plus/") for line in lines]
            self.img_label = [int(line.strip().split("*")[-2]) for line in lines]
            # 回归改为分类
            self.split_y1 = [filter_cls(int(float(line.strip().split("*")[-1].split("_")[0])*32)) for line in lines]
            self.split_y2 = [filter_cls(int(float(line.strip().split("*")[-1].split("_")[1])*32)) for line in lines]
            #self.split_y = [list(map(float,line.strip().split("*")[-1].split("_"))) for line in lines]
            #self.split_y = torch.Tensor(self.split_y).float()
            #print(self.img_label)
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)
        split_y1 = self.split_y1[item]
        split_y2 = self.split_y2[item]
#         if "double" in img_name:
#             split_y = 0.4
#         else:
#             split_y = 0.0
        
        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))
        #print(img, label,img_name,split_y)
        return img, label,img_name,split_y1,split_y2

def train_model(model, criterion, optimizer, scheduler, num_epochs, use_gpu):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_acc_v1 = 0.0
    best_acc_v2 = 0.0

    for epoch in range(num_epochs):
        begin_time = time.time()
        count_batch = 0
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            count_batch = 0
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_corrects_v1 = 0
            running_corrects_v2 = 0

            # Iterate over data.
            for data in dataloders[phase]:
                count_batch += 1
                # get the inputs
                inputs, labels,_ ,split_y1,split_y2 = data
                #print(labels)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                    split_y1 = Variable(split_y1.cuda())
                    split_y2 = Variable(split_y2.cuda())
                else:
                    inputs, labels ,split_y1,split_y2 = Variable(inputs), Variable(labels),Variable(split_y1),Variable(split_y2)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                #print(inputs, labels ,split_y)
                outputs,vc1,vc2 = model(inputs)
                outputs = torch.squeeze(outputs)
                vc1 = torch.squeeze(vc1)
                vc2 = torch.squeeze(vc2)
                #print("###############-output-sc",outputs)
                #print("###############-output-vc",vc)
                _, preds = torch.max(outputs.data, 1)
                _1, preds_vc1 = torch.max(vc1.data, 1)                
                _2, preds_vc2 = torch.max(vc2.data, 1)
                #print(split_y1,split_y2)
               # print(outputs.shape,vc.shape, labels.shape ,split_y.shape)
                loss = criterion(outputs,vc1,vc2, labels ,split_y1,split_y2)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.data[0]
                
                running_corrects += torch.sum(preds == labels.data)
                running_corrects_v1 += torch.sum(preds_vc1 == split_y1.data)
                running_corrects_v2 += torch.sum(preds_vc2 == split_y2.data)
                
                # print result every 10 batch
                if count_batch%10 == 0:
                    batch_loss = running_loss / (batch_size*count_batch)
                    batch_acc = running_corrects.item() / (batch_size*count_batch)
                    batch_acc_v1 = running_corrects_v1.item() / (batch_size*count_batch)
                    batch_acc_v2 = running_corrects_v2.item() / (batch_size*count_batch)
                    #print(running_corrects.item(),batch_size,count_batch,batch_acc)
                    print('{} Epoch [{}] Batch [{}] Loss: {:.6f} cls_Acc: {:.6f}  v1_Acc: {:.6f}  v2_Acc: {:.6f} Time: {:.6f}s'. \
                          format(phase, epoch, count_batch, batch_loss, batch_acc, batch_acc_v1,batch_acc_v2,time.time()-begin_time))
                    begin_time = time.time()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]
            epoch_acc_v1 = running_corrects_v1.item() / dataset_sizes[phase]
            epoch_acc_v2 = running_corrects_v2.item() / dataset_sizes[phase]
           # print(running_corrects.item(),dataset_sizes[phase],epoch_acc)
            print('{} Loss: {:.6f} cls_Acc: {:.6f} v1_Acc: {:.6f}  v2_Acc: {:.6f}'.format(phase, epoch_loss, epoch_acc,epoch_acc_v1,epoch_acc_v2))

            # save model
            if phase == 'train':
                if not os.path.exists('output'):
                    os.makedirs('output')
                torch.save(model, 'output/resnet_epoch{}.pkl'.format(epoch))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_acc_v1 = epoch_acc_v1
                best_acc_v2 = epoch_acc_v2
                best_model_wts = model.state_dict()

            
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val cls_Acc: {:6f}  v1_Acc: {:6f}  v2_Acc: {:6f}'.format(best_acc,best_acc_v1,best_acc_v2))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


import Augmentor
from Augmentor.Operations import *



if __name__ == '__main__':
    # 设置增强方式
    p = Augmentor.Pipeline()
    eras = RandomErasing(probability=0.7, rectangle_area=0.1)#min_factor=0, max_factor=1)
    p.add_operation(eras)
    bright = RandomBrightness(probability=0.4, min_factor=0.4, max_factor=1.5) #(probability=0.7, rectangle_area=0.1)#min_factor=0, max_factor=1)
    p.add_operation(bright)
    color = RandomColor(probability=0.4, min_factor=0.5, max_factor=0.8) #(probability=0.7, rectangle_area=0.1)#min_factor=0, max_factor=1)
    p.add_operation(color)
    contrast = RandomContrast(probability=0.4, min_factor=0.5, max_factor=0.9) #(probability=0.7, rectangle_area=0.1)#min_factor=0, max_factor=1)
    p.add_operation(contrast)
# 定义一个类，需要创建模型的时候，就实例化一个对象

    data_transforms = {
        'train': transforms.Compose([
            #transforms.CenterCrop((128,256)),
            transforms.Resize((64,64)),
            p.torch_transform(),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((64,64)),
            #transforms.Scale(256),
            #transforms.CenterCrop((256,244)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    use_gpu = torch.cuda.is_available()

    batch_size = 16
    num_class = 2

#     image_datasets = {x: customData(img_path='/ImagePath',
#                                     txt_path=('./txt_file/train_0w_0108/' + x + '.txt'),
#                                     data_transforms=data_transforms,
#                                     dataset=x) for x in ['train', 'val']}

#     # wrap your data and label into Tensor
#     dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
#                                                  batch_size=batch_size,
#                                                  shuffle=True) for x in ['train', 'val']}

#     dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

#     #get model and replace the original fc layer with your fc layer
#     #model_ft = models.resnet152(pretrained=False)
    
#     #num_ftrs = model_ft.fc.in_features
    
#     #model_ft.fc = nn.Linear(num_ftrs, num_class)

#     model_ft = jmodel #torch.load('output/best_resnet_152_split_4point_aug_0108_20angle_10plus_3channle_dp02_0w_aug_reg_to_cls_02_fc.pkl') #jmodel #
# #     model_ft = models.vgg19_bn()
# #     num_ftrs = model_ft.classifier[6].in_features
# #     model_ft.classifier[6] = nn.Linear(num_ftrs, num_class)
    
#     # if use gpu
#     if use_gpu:
#         model_ft = model_ft.cuda()

#     # define cost function
#     criterion = JamieSplitPlateLoss() # nn.CrossEntropyLoss()

#     # Observe that all parameters are being optimized
#     #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.000001, momentum=0.9)
#     optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.01 , betas=(0.5, 0.999))
#     # Decay LR by a factor of 0.2 every 5 epochs
#     exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.2)
#     #exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, 'min',factor=0.5, patience=2, verbose=True)
    
#     # Decay LR by a factor of 0.2 every 5 epochs
#     #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.2)

#     #multi-GPU
#     model_ft = torch.nn.DataParallel(model_ft, device_ids=[0])

#     #train model
#     model_ft = train_model(model=model_ft,
#                            criterion=criterion,
#                            optimizer=optimizer_ft,
#                            scheduler=exp_lr_scheduler,
#                            num_epochs=40,
#                            use_gpu=use_gpu)
#     # vbest_resnet_152_split_4point_aug_0108_20angle_10plus_3channle_dp02_0w_aug_reg_to_cls_01 第一次 分类 
#     #save best model
#     torch.save(model_ft,"output/best_resnet_18_split_0113_01.pkl")
