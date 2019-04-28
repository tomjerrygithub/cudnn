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

#定义一个类，需要创建模型的时候，就实例化一个对象

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
        flatten = x.view(x.size(0), -1)
        flatten = self.out(flatten)
        vertical_coordinate1 = self.vertical_coordinate1(flatten)
        vertical_coordinate2 = self.vertical_coordinate2(flatten)
        score = self.score(flatten)

        return score,vertical_coordinate1,vertical_coordinate2

model = JamieSplitPlateModel()
#model = resnet50(args)
#weights = torch.load('params.pkl')
#model.load_state_dict(weights)
model = torch.load('best_resnet_18_0329_finetune_nopara_nospecilgpu.pkl')
#model.eval()
#model.load_state_dict(weights)
dummy_input = Variable(torch.randn(1, 3, 224, 224))
 #jmodel # torchvision.models.resnet18(pretrained=False).cuda()
# model(dummy_input)
# torch.save(model,"vcolor.pkl")

torch.save(model.state_dict(), 'vcolor22.pkl')
weights = torch.load('vcolor22.pkl')

from collections import OrderedDict
new_state_dict = OrderedDict()
for k,v in weights.items():
    name = k[21:]
    new_state_dict[name] = v
model.cuda()    
model.load_state_dict(new_state_dict,strict=False)
#output_names=["score","vertical_coordinate1","vertical_coordinate2"]
#device = torch.device("cuda:1")
model.eval()
model.cuda()

# 向模型中输入数据以得到模型参数 
example = torch.rand(1,3,64, 280).cuda() 
traced_script_module = torch.jit.trace(model,example)
 
# 保存模型
traced_script_module.save("split22.pt")
