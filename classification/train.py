import time
import numpy as np
import torch
import torch.nn as nn
import  torch.nn.functional as F
import os
import paraset
from torchvision import  transforms
import  matplotlib.pyplot as plt
import  basic
import os

loadfiles = '/media/josen/data/opencv_learn/data'
batch_size = 70
num_epoch = 10
num_work = 2
num_classes = 10
gpu = 'cuda:0'
learning_rate = 0.0001
if torch.cuda.is_available():
    device = torch.device(gpu)
else :
    device = torch.device('cpu')

#transform = transforms.Compose([transforms.Resize((240,240)),transforms.RandomCrop((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
transform = transforms.Compose([transforms.Resize((380,380)),transforms.RandomCrop((299,299)),transforms.ToTensor()])
#, transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

#  Normaliztion For RGB mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]
print('Load Devices Type %s ----%d'%(device.type,device.index))
Data = paraset.GetTestData(loadfiles,batch_size,num_work,transform,'carf10')
train_loader , valid_loader , test_loader = Data.GetResize(index=1)
torch.manual_seed(1)
# resnet 18 Total Training Time : 170.93
for r,t in test_loader:
    print(r.shape,t.shape)
    break
'''
Epoch 050|050 Batch:128|391 |Cost:0.006
Epoch :050, train_acc:99.470,valid_acc:99.600,test_acc:86.440, Spending 291.307 min
Total Training Time : 291.31
Epoch :050, train_acc:99.308,valid_acc:99.400,test_acc:86.200, Spending 552.336 min
Total Training Time : 552.34
Epoch :050, train_acc:99.178,valid_acc:99.600,test_acc:79.740, Spending 198.194 min
Total Training Time : 198.19
'''
#resnet34 = paraset.ResNet101(paraset.Bottleneck,layers=[3, 4, 24, 3],num_classes=num_classes,grayscale=False)
#print("ResNet  50 Loading--------->here")
#model = resnet34.to(device)
model = basic.InceptionV3(num_classes)
model = model.to(device)


optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
ClassFic = paraset.TrainClassification(optim,num_epoch,batch_size,num_classes,train_loader,valid_loader,test_loader,model,device)
#ClassFic.getSummary(shape=(3,299,299))
ClassFic.Load_pretraining('./model/InceptionV3-20_epoch-pt')
ClassFic.train(index=1000)
ClassFic.DrawTrain("InceptionV3")
torch.save(model.state_dict(),'./model/InceptionV3-%d_epoch-pt'%(num_epoch))
print("Save Model ")
