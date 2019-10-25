import os
import numpy as np
import torch
import torch.nn as nn
import  time
from torch.utils.data import  DataLoader
from torchvision import  transforms
from torchvision import  datasets
from torch.utils.data import  Subset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchsummary import  summary
class GetXminist():
    def __init__(self,cuda='cuda:0'):
        self.cuda = cuda
        self.loadfile = '/media/josen/data/opencv_learn/data'
        self.learning_rate = 0.005
        self.randomset = 123
        self.num_epoch = 3
        self.batch_size = 256
        self.num_features = 28*28
        self.num_hidden_1 = 32
        self.classes = 10
class CNNmnist():
    def __init__ (self,cuda='cuda:0'):
        self.cuda = cuda
        self.loadfile =  '/media/josen/data/opencv_learn/data'
        self.learning_rate = 0.001
        self.num_epoch = 100
        self.batch_size = 128 * 2
        self.num_classes = 10
        self.num_features = 28*28
        self.num_latent  = 50
        self.ave_hidden_1 = 500
        self.ave_hidden_2 = 35
#### cuda :0 Total Training Time: 10.13 min
### cuda:1  Total Training Time: 9.56 min

def one_hot(labels,num_classes,device):
      ##
      #  labels : tensor
      #  num_classes : number of labels
        label_onehot = torch.zeros(labels.size()[0] ,num_classes ).to(device)
        label_onehot.scatter_(1, labels.view(-1,1),1 )
        return label_onehot

train_index = torch.arange(0,49000)
valid_index = torch.arange(49000,50000)
class GetTestData():
    def __init__(self, loadfile , batch_size , num_work ,trainform,mode='carf10'):
        self.load_files = loadfile
        self.batch_size = batch_size
        if num_work>0:
            self.num_work = num_work
        else :
            self.num_work = 0
        self.mode = mode
        self.transform =trainform
        ##
        ##
    def GetRawData(self,index=1):
        if self.mode == 'carf10' :
            train_data = datasets.CIFAR10(self.load_files,train=True,transform=self.transform.ToTensor())
            test_data = datasets.CIFAR10(self.load_files,train=False,transform=self.transform.ToTensor())
        elif self.mode == 'minist':
            train_data = datasets.MNIST(self.load_files,train=True,transform=self.transform.ToTensor())
            test_data = datasets.MNIST(self.load_files,train=False,transform=self.transform.ToTensor())
        else :
            train_data = datasets.FashionMNIST(self.load_files,train=True,transform=self.transform.ToTensor())
            test_data = datasets.FashionMNIST(self.load_files,train=False,transform=self.transform.ToTensor())
        if index == 1:
            train_dataset = Subset(train_data,train_index)
            valid_dataset = Subset(train_data,valid_index)
            train_loader = DataLoader(train_data,batch_size=self.batch_size,shuffle=True,num_workers=self.num_work)
            valid_loader = DataLoader(valid_dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_work)
            test_loader = DataLoader(test_data,batch_size=self.batch_size,shuffle=False,num_workers=self.num_work)
            return train_loader , valid_loader , test_loader
        else :
            train_loader = DataLoader(train_data,batch_size=self.batch_size,shuffle=True,num_workers=self.num_work)
            test_loader = DataLoader(test_data,batch_size=self.batch_size,shuffle=False,num_workers=self.num_work)
            return train_loader , test_loader
    def GetResize(self,index = 1):
        if self.mode == 'carf10' :
            train_data = datasets.CIFAR10(self.load_files,train=True,transform=self.transform)
            test_data = datasets.CIFAR10(self.load_files,train=False,transform=self.transform)
        elif self.mode == 'minist':
            train_data = datasets.MNIST(self.load_files,train=True,transform=self.transform)
            test_data = datasets.MNIST(self.load_files,train=False,transform=self.transform )
        else :
            train_data = datasets.FashionMNIST(self.load_files,train=True,transform=self.transform )
            test_data = datasets.FashionMNIST(self.load_files,train=False,transform=self.transform)
        if index == 1:
            train_dataset = Subset(train_data,train_index)
            valid_dataset = Subset(train_data,valid_index)
            train_loader = DataLoader(train_data,batch_size=self.batch_size,shuffle=True,num_workers=self.num_work)
            valid_loader = DataLoader(valid_dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_work)
            test_loader = DataLoader(test_data,batch_size=self.batch_size,shuffle=False,num_workers=self.num_work)
            return train_loader , valid_loader , test_loader
        else :
            train_loader = DataLoader(train_data,batch_size=self.batch_size,shuffle=True,num_workers=self.num_work)
            test_loader = DataLoader(test_data,batch_size=self.batch_size,shuffle=False,num_workers=self.num_work)
            return train_loader , test_loader




#### Written by Josenxiao
def conv3x3(inplane , outplane, stride=1):
    return nn.Conv2d(inplane,outplane,kernel_size=3,stride=stride,padding=1)
## ZhiwenXiao
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class ResNet101(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet101, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=0)
        #self.fc = nn.Linear(2048 * block.expansion, num_classes)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale=False):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

def evaluteTop5(model, loader,device):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x,y = x.to(device),y.to(device)
        with torch.no_grad():
            logits = model(x)
            maxk = max((1,5))
            y_resize = y.view(-1,1)
            _, pred = logits.topk(maxk, 1, True, True)
            correct += torch.eq(pred, y_resize).sum().float().item()
    return correct / total


class TrainClassification():
    #
    #  That Class is universial API for classifiaction
    def __init__(self, optimizer,num_epoch,batch_size,num_classes,train_loader,valid_loader,test_loader,model,device):
        self.optim = optimizer
        self.num_classes = num_classes
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.model = model
        self.device = device
        self.train_list = []
        self.valid_list = []
        self.test_list = []
    def train(self, index=200):
        start_time = time.time()
        for epoch in range(self.num_epoch):
            self.model.train()
            for batch_idx , (features , targests ) in enumerate(self.train_loader):
                features = features.to(self.device)
                targests = targests.to(self.device)
                logit , pro = self.model(features)
                cost = F.cross_entropy(logit,targests)
                self.optim.zero_grad()
                cost.backward()
                self.optim.step()
                if not batch_idx % index  and batch_idx >0 :
                    print("Epoch %03d|%03d Batch:%03d|%03d |Cost:%.3f"%(epoch+1,self.num_epoch,self.batch_size,len(self.train_loader),cost))
            self.model.eval()
            with torch.set_grad_enabled(False):
                train_acc = self.compute_accuracy(self.model,self.train_loader,self.device)
                valid_acc = self.compute_accuracy(self.model,self.valid_loader,self.device)
                self.train_list.append(train_acc)
                self.valid_list.append(valid_acc)
            self.model.eval()
            with torch.set_grad_enabled(False):
                test_acc = self.compute_accuracy(self.model,self.test_loader,self.device)
                #print("Total Training eplased %.3fmin , test_acc: %.3f"%(Eplased,test_acc))
                self.test_list.append(test_acc)
            Eplased = (time.time() - start_time) /60
            print("Epoch :%03d| Total Epoch: %03d, train_acc:%.3f,valid_acc:%.3f,test_acc:%.3f, Spending %.3f min"%(epoch+1,self.num_epoch,train_acc,valid_acc,test_acc,Eplased))
        Eplased = (time.time() - start_time) /60
        print("Total Training Time : %.2f"%(Eplased))
    def compute_accuracy(self,model,data_loader,device):
        correct_preb , num_example = 0 , 0
        for i, (featrues , targets) in enumerate( data_loader):
            featrues = featrues.to(device)
            targets = targets.to(device)
            logit , prob = model(featrues)
            _ , predicted = torch.max(prob,1)
            num_example += targets.size(0)
            correct_preb += ( predicted==targets ).sum()
        return correct_preb.float() / num_example * 100
    ##############################
    def get_parameter_number(self):
        net = self.model
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    def DrawTrain(self,name='ResNet101/Batch'):
        plt.plot(range(0,len(self.train_list)),self.train_list ,label='Training')
        plt.plot(range(0,len(self.valid_list)),self.valid_list,label='Valiation')
        plt.plot(range(0,len(self.test_list)),self.test_list,label='TestAcc')
        plt.title(name)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
    def Load_pretraining(self,files ):
        if os.path.isfile(files):
            self.model.load_state_dict(torch.load(files) )
            print("Load pretraining:",files)
        else:
            print("No Files Pretraining")
    def getSummary(self,shape=(3,224,244)):
        summary(self.model,shape)
