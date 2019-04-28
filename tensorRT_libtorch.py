from torch.autograd import Variable
import torch.onnx
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms
import torch.nn.functional as F
import time
import os,cv2
from torch.utils.data import Dataset
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
        model_ft = models.resnet18(pretrained=False)
    
        num_final_in = model_ft.fc.in_features
        
        # edit 0109   1 channel
       # model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        model_ft.avgpool = nn.AvgPool2d(kernel_size=(3,3), stride=1, padding=1)    # kernel_size=(1,7), stride=3           

        # 防止过拟合
        model_ft.dp = nn.Dropout(p=0.5)


        self.backbone = nn.Sequential(*list(model_ft.children())[0:9])

        self.out = nn.Linear(in_features=9216, out_features=512, bias=True)
             
        #self.vertical_coordinate1 = nn.Conv2d(512, 32, 1)
        self.vertical_coordinate1 = nn.Linear(in_features=512 , out_features=32, bias=True)
        #self.vertical_coordinate2 = nn.Conv2d(512, 32, 1)
        self.vertical_coordinate2 = nn.Linear(in_features=512 , out_features=32, bias=True)
        self.score = nn.Linear(in_features=512 , out_features=2, bias=True) #nn.Conv2d(512, 2, 1)
        
    
    
    def forward(self,x):
        x = self.backbone(x)
        flatten = x.view(len(x), -1)
        flatten = self.out(flatten)
        vertical_coordinate1 = self.vertical_coordinate1(flatten)
        vertical_coordinate2 = self.vertical_coordinate2(flatten)
        score = self.score(flatten)

        return score,vertical_coordinate1,vertical_coordinate2

model = JamieSplitPlateModel()
#model = resnet50(args)
#weights = torch.load('params.pkl')
#model.load_state_dict(weights)
model = torch.load('output/best_resnet_18_0329_finetune_nopara.pkl')
model.eval()
#model.load_state_dict(weights)
#dummy_input = Variable(torch.randn(10, 3, 64, 280))
 #jmodel # torchvision.models.resnet18(pretrained=False).cuda()
#model(dummy_input)
#torch.save(model,"test.pkl")
#weights = torch.load('output/best_resnet_18_split_0118_dp07_aug_01_finetune_nopara.pkl')

torch.save(model.state_dict(), 'split_nopara.pkl')
weights = torch.load('split_nopara.pkl')
from collections import OrderedDict
new_state_dict = OrderedDict()
for k,v in weights.items():
    name = k[:]
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
#output_names=["score","vertical_coordinate1","vertical_coordinate2"]
cuda1 = torch.device('cuda:0')
model.cuda()
model.eval()

# 向模型中输入数据以得到模型参数 
example = torch.rand(1,3,224, 224).cuda() 
traced_script_module = torch.jit.trace(model,example)
 
# 保存模型
traced_script_module.save("split_1.pt")



###
### 多卡上跑pytorch 怎样操作？
###
#!/usr/bin/python3
# coding: utf-8
import torch
from torchvision import datasets, transforms
import torchvision
from tqdm import tqdm

device_ids = [3, 4, 6, 7]
BATCH_SIZE = 64

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
data_train = datasets.MNIST(root = "./data/",
                            transform=transform,
                            train = True,
                            download = True)
data_test = datasets.MNIST(root="./data/",
                           transform = transform,
                           train = False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                # 这里注意batch size要对应放大倍数
                                                batch_size = BATCH_SIZE * len(device_ids), 
                                                shuffle = True,
                                                 num_workers=2)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size = BATCH_SIZE * len(device_ids),
                                               shuffle = True,
                                                num_workers=2)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(stride=2, kernel_size=2),
    )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 128, 1024),                            
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10)
    )
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


model = Model()
model = torch.nn.DataParallel(model, device_ids=device_ids) # 声明所有可用设备
model = model.cuda(device=device_ids[0])  # 模型放在主设备

cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

n_epochs = 50
for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch, n_epochs))
    print("-"*10)
    for data in tqdm(data_loader_train):
        X_train, y_train = data
        # 注意数据也是放在主设备
        X_train, y_train = X_train.cuda(device=device_ids[0]), y_train.cuda(device=device_ids[0])
        
        outputs = model(X_train)
        _,pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs, y_train)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.data.item()
        running_correct += torch.sum(pred == y_train.data)
    testing_correct = 0
    for data in data_loader_test:
        X_test, y_test = data
        X_test, y_test = X_test.cuda(device=device_ids[0]), y_test.cuda(device=device_ids[0])
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)
    print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss/len(data_train),
                                                                                      100*running_correct/len(data_train),
                                                                                      100*testing_correct/len(data_test)))
torch.save(model.state_dict(), "model_parameter.pkl")
