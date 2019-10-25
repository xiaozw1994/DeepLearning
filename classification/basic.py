import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#
#     set Image from [0~255] to [0,1] uses tranforms.Compose([tranforms.ToTensor()])
#    make Image from [0~255] to [-1,1] use transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
class VGG16(nn.Module):
    def __init__(self,num_classes,grayscale=False):
        dim = None
        if grayscale==True:
            dim = 1
        else :
            dim = 3

        super(VGG16,self).__init__()
        self.vgg_bone = nn.Sequential(
          nn.Conv2d(dim,64,kernel_size=3,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(64,64,kernel_size=3,padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
          nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(128,128,kernel_size=3, padding=1 ),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
          nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
          nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
        )
        self.vgg_logit = nn.Sequential(
        nn.Linear(7*7*512,4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096,4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096,num_classes),
        )
        for m in self.modules():
            if isinstance(m ,nn.Conv2d ):
                m.weight.detach().normal_(0,0.05)
                if m.bias is not None :
                    m.bias.data.detach().zero_()
            elif isinstance(m,nn.Linear):
                m.weight.detach().normal_(0,0.05)
                m.bias.detach().detach().zero_()
    def forward(self,x):
        x =self.vgg_bone(x)
        x =x.view(x.size(0),-1)
        logit  = self.vgg_logit(x)
        prob = F.softmax(logit)
        return  logit,prob
####################################################
class VGG19(nn.Module):
    def __init__(self,num_classes,grayscale=False):
        dim = None
        if grayscale==True:
            dim = 1
        else :
            dim = 3
        super(VGG19,self).__init__()
        self.vgg_bone = nn.Sequential(
          nn.Conv2d(dim,64,kernel_size=3,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(64,64,kernel_size=3,padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
          nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(128,128,kernel_size=3, padding=1 ),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
          nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
          nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
           nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
           nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
        )
        self.vgg_logit = nn.Sequential(
        nn.Linear(7*7*512,4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096,4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096,num_classes),
        )
        for m in self.modules():
            if isinstance(m ,nn.Conv2d ):
                m.weight.detach().normal_(0,0.05)
                if m.bias is not None :
                    m.bias.data.detach().zero_()
            elif isinstance(m,nn.Linear):
                m.weight.detach().normal_(0,0.05)
                m.bias.detach().detach().zero_()
    def forward(self,x):
        x =self.vgg_bone(x)
        x =x.view(x.size(0),-1)
        logit  = self.vgg_logit(x)
        prob = F.softmax(logit)
        return  logit,prob



#######################################
class MobleNetV1(nn.Module):
        def __init__(self,num_classes,grayscale=False):
            dim = None
            if grayscale == True:
                dim = 1
            else :
                dim = 3
            super(MobleNetV1,self).__init__()
            self.mobilebone = nn.Sequential(
            self._conv_bn(dim,32,2),
            self._conv_dw(32,64,1),
            self._conv_dw(64,128,2),
            self._conv_dw(128,128,1),
             self._conv_dw(128,256,2),
             self._conv_dw(256,256,1),
             self._conv_dw(256,512,2),
             self._top_conv(512,512,5),
             self._conv_dw(512,1024,2),
             self._conv_dw(1024,1024,1),
            )
            self.avgpool = nn.AvgPool2d(kernel_size=7,stride=1)
            self.fc = nn.Linear(1024,num_classes)
            for m in self.modules():
                if isinstance(m,nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, (2. / n)**.5)
                if isinstance(m,nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        def forward(self,x):
            x = self.mobilebone(x)
            x = self.avgpool(x)
            x =x.view(x.size(0),-1)
            x = self.fc(x)
            prob = F.softmax(x)
            return x , prob
        def _top_conv(self,in_channel,out_channel,blocks):
            layers = []
            for i in range(blocks):
                layers.append(self._conv_dw(in_channel,out_channel,1))
            return nn.Sequential(*layers)
        def _conv_bn (self,in_channel,out_channel,stride):
            return  nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,stride,padding=1, bias=False ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            )
        def _conv_dw(self,in_channel,out_channel,stride):
            return nn.Sequential(
            nn.Conv2d(in_channel,in_channel,3,stride,1, groups=in_channel,bias=False ),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel,out_channel,1,1,0,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False),
            )

class BaseMobileResNet(nn.Module):
    def __init__(self,in_channel,out_channel,stride,ratio=1):
        super(BaseMobileResNet,self).__init__()
        self.stride = stride
        self.res = self.stride==1 and in_channel == out_channel
        self.features = nn.Sequential(
        nn.Conv2d(in_channel,in_channel*ratio,1,1,0,bias=False),
        nn.BatchNorm2d(in_channel*ratio),
        nn.ReLU6(inplace=True),
        ###
        nn.Conv2d(in_channel*ratio,in_channel*ratio,3,stride, 1,groups=in_channel*ratio,bias=False),
        nn.BatchNorm2d(in_channel*ratio),
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_channel*ratio,out_channel,1,1,0,bias=False),
        nn.BatchNorm2d(out_channel),
        )
    def forward(self,x):
        if self.res :
            return x+ self.features(x)
        else:
            return self.features(x)


class MobileNetV2(nn.Module):
    def __init__(self,num_classes,bottle=BaseMobileResNet,layers=[1,2,3,4,3,3,1],ratio=[1,6,6,6,6,6,6],strides=[1,2,2,2,1,2,1]):
        self.layers = layers
        self.bottle = bottle
        super(MobileNetV2,self).__init__()
        self.conv_bn = self._conv_bn(3,32,2)
        self.layer1 = self._make_layer(32,16,layers[0],ratio[0],strides[0])
        self.layer2 = self._make_layer(16,24,layers[1],ratio[1],strides[1])
        self.layer3 = self._make_layer(24,32,layers[2],ratio[2],strides[2])
        self.layer4 = self._make_layer(32,64,layers[3],ratio[3],strides[3])
        self.layer5 = self._make_layer(64,96,layers[4],ratio[4],strides[4])
        self.layer6 = self._make_layer(96,160,layers[5],ratio[5],strides[5])
        self.layer7 = self._make_layer(160,320,layers[6],ratio[6],strides[6])
        self.bottom = self._conv1x1(320,1280)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Sequential(
        nn.Dropout(p=0.8),
        nn.Linear(1280,num_classes),
        )
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] *m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5 )
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()
    def forward(self,x):
        x = self.conv_bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.bottom(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        pro = F.softmax(x)
        return x,pro

    def _make_layer(self,in_channel,out_channel,layer,ratio,stride):
        cnn = []

        if layer >1 :
            for i in range(layer):
                out_channel = int(out_channel * 1)
                if i==0:
                    cnn.append( self.bottle(in_channel,out_channel,stride,ratio ) )
                else:
                    cnn.append(self.bottle(in_channel,out_channel,1,ratio))
                in_channel = out_channel
        elif layer==1:
            cnn.append(self.bottle(in_channel,int(out_channel*1),1,ratio))
        return nn.Sequential(*cnn)

    def _conv_bn(self,in_channel,out_channel,stride):
        return nn.Sequential(
        nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1, bias=False ),
        nn.BatchNorm2d(out_channel),
        nn.ReLU6(inplace=True),
        )
    def _conv1x1(self,in_channel,out_channel,stride=1):
        return nn.Sequential(
        nn.Conv2d(in_channel,out_channel,1,1,0,bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU6(inplace=True),
        )

"""
Reviving The googlenet by pytorch

Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
    http://arxiv.org/pdf/1409.4842v1.pdf.
"""

class InceptionV1_base(nn.Module):
    def __init__(self, in_channel, layers=[64,96,128,16,32,32]):
        super(InceptionV1_base,self).__init__()
        self.branch_1 = nn.Sequential(
        nn.Conv2d(in_channel,layers[0],kernel_size=1,bias=False),
        nn.ReLU6(inplace=True),
        )
        self.branch_2 = nn.Sequential(
        nn.Conv2d(in_channel,layers[1],kernel_size=1,bias=False),
        nn.ReLU6(inplace=True),
        nn.Conv2d(layers[1],layers[2],kernel_size=3,stride=1,padding=1,bias=False),
        nn.ReLU6(inplace=True),
        )
        self.branch_3 = nn.Sequential(
        nn.Conv2d(in_channel,layers[3],kernel_size=1,stride=1,padding=0,bias=False),
        nn.ReLU6(inplace=True),
        nn.Conv2d(layers[3],layers[4],kernel_size=3,stride=1,padding=1,bias=False),
        nn.ReLU6(inplace=True),
        )
        self.branch_4 = nn.Sequential(
        nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
        nn.Conv2d(in_channel,layers[5],kernel_size=1,stride=1,padding=0,bias=False),
        nn.ReLU6(inplace=True),
        )
    def forward(self,x):
        b_1 = self.branch_1(x)
        b_2 = self.branch_2(x)
        b_3 = self.branch_3(x)
        b_4 = self.branch_4(x)
        y = torch.cat([b_1,b_2,b_3,b_4],dim=1)
        return y

class InceptionV1(nn.Module):
    def __init__(self,num_class,block=InceptionV1_base,grayscale=False):
        if grayscale:
            dim = 1
        else :
            dim = 3
        self.block = block
        super(InceptionV1,self).__init__()
        self.bottle = nn.Sequential(
        nn.Conv2d(dim,64,kernel_size=7,stride=2,padding=3,bias=False),
        nn.ReLU6(inplace=True),
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        nn.Conv2d(64,64,kernel_size=1,stride=1,padding=0,bias=False),
        nn.ReLU6(inplace=True),
        nn.Conv2d(64,192,kernel_size=3,stride=1,padding=1,bias=False),
        nn.ReLU6(inplace=True),
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )
        self.layer1 = self._make_layer(192,[64,96,128,16,32,32  ])
        self.layer2 = self._make_layer(256,[128,128,192,32,96,64])
        self.max = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer3 = self._make_layer(480,[192,96,208,16,48,64])
        self.layer4 = self._make_layer(512,[160,112,224,24,64,64])
        self.layer5 = self._make_layer(512,[128,128,256,24,64,64])
        self.layer6 = self._make_layer(512,[112,144,288,32,64,64])
        self.layer7 = self._make_layer(528,[256,160,320,32,128,128])
        self.layer8 = self._make_layer(832,[256,160,320,32,128,128])
        self.layer9 = self._make_layer(832,[384,192,384,48,128,128])
        self.avg = nn.AvgPool2d(7,stride=1)
        self.bottom = nn.Sequential(
        nn.Dropout(p=0.8),
        nn.Linear(1024,num_class),
        )
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] *m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5 )
            elif  isinstance(m,nn.Linear):
                m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()

    def _make_layer(self, in_channel,layers ):
        blocks = []
        blocks.append( self.block(in_channel,layers) )
        return nn.Sequential(*blocks)
    def forward(self,x):
        x = self.bottle(x)
        x =self.layer1(x)
        x =self.layer2(x)
        x = self.max(x)
        x =self.layer3(x)
        x =self.layer4(x)
        x =self.layer5(x)
        x =self.layer6(x)
        x = self.max(x)
        x =self.layer7(x)
        x =self.layer8(x)
        x =self.layer9(x)
        x = self.avg(x)
        x = x.view(x.size(0),-1 )
        x = self.bottom(x)
        pro = F.softmax(x)
        return x , pro
"""
InceptionV2
Constructs an Inception v2 network from inputs to the given final endpoint.
  This method can construct the network up to the layer inception(5b) as
  described in http://arxiv.org/abs/1502.03167.
"""

class InceptionV2_base(nn.Module):
    def __init__(self,in_channel,layers=[64,64,64,64,96,96,32]):
        super( InceptionV2_base,self).__init__()
        self.branch_1 = nn.Sequential(
        nn.Conv2d(in_channel,layers[0],kernel_size=1,stride=1,padding=0,bias=False),
        nn.ReLU6(inplace=True),
        )
        self.branch_2 = nn.Sequential(
        nn.Conv2d(in_channel,layers[1],kernel_size=1,stride=1,padding=0,bias=False),
        nn.ReLU6(inplace=True),
        nn.Conv2d(layers[1],layers[2],kernel_size=3,stride=1,padding=1,bias=False),
        nn.ReLU6(inplace=True),
        )
        self.branch_3 = nn.Sequential(
        nn.Conv2d(in_channel,layers[3],kernel_size=1,stride=1,padding=0,bias=False),
        nn.ReLU6(inplace=True),
        nn.Conv2d(layers[3],layers[4],kernel_size=3,stride=1,padding=1,bias=False),
        nn.ReLU6(inplace=True),
        nn.Conv2d(layers[4],layers[5],kernel_size=3,stride=1,padding=1,bias=False),
        nn.ReLU6(inplace=True),
        )
        self.branch_4 = nn.Sequential(
        nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
        nn.Conv2d(in_channel,layers[6],1,stride=1,padding=0,bias=False),
        nn.ReLU6(inplace=True),
        )
    def forward(self,x):
        b_1 = self.branch_1(x)
        b_2 = self.branch_2(x)
        b_3 = self.branch_3(x)
        b_4 = self.branch_4(x)
        y = torch.cat([b_1,b_2,b_3,b_4],dim=1)
        return y

class Inceptionv2_dicount(nn.Module):
    def __init__(self,in_channel,layers=[128,160,64,96,96]):
        super(Inceptionv2_dicount,self).__init__()
        self.branch1 = nn.Sequential(
         nn.Conv2d(in_channel,layers[0],1,stride=1,padding=0,bias=False),
         nn.ReLU6(inplace=True),
         nn.Conv2d(layers[0],layers[1],3, stride=2,padding=1,bias=False),
         nn.ReLU6(inplace=True),
        )
        self.branch2 = nn.Sequential(
        nn.Conv2d(in_channel,layers[2],1,stride=1,padding=0,bias=False),
        nn.ReLU6(True),
        nn.Conv2d(layers[2],layers[3],3,stride=1,padding=1,bias=False),
        nn.ReLU6(inplace=True),
        nn.Conv2d(layers[3],layers[4],3,stride=2,padding=1,bias=False),
        nn.ReLU6(inplace=True),
        )
        self.branch3 = nn.Sequential(
        nn.MaxPool2d(3,stride=2,padding=1),
        )
    def forward(self,x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1,b2,b3],dim=1)


class InceptionV2(nn.Module):
    def __init__(self,num_class,block=InceptionV2_base,strideblock=Inceptionv2_dicount ,grayscale=False):
        self.block = block
        self.discount = strideblock
        if grayscale:
            dim = 1
        else:
            dim = 3
        super(InceptionV2,self).__init__()
        self.bottle=nn.Sequential(
        nn.Conv2d(dim,64,7,stride=2,padding=3,bias=False),
        nn.ReLU6(True),
        nn.MaxPool2d(3,stride=2,padding=1),
        nn.Conv2d(64,64,1,1,0,bias=False),
        nn.ReLU6(inplace=True),
        nn.Conv2d(64,192,3,1,padding=1,bias=False),
        nn.ReLU6(inplace=True),
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )
        self.layer1 = self._make_layer(192,[ 64,64,64,64,96,96,32 ])
        self.layer2 = self._make_layer(256,[64,64,96,64,96,96,64 ])
        self.layer3 = nn.Sequential(strideblock(320))
        self.layer4 = self._make_layer(576,[224,64,96,96,128,128,128])
        self.layer5 = self._make_layer(576,[192,96,128,96,128,128,128])
        self.layer6 = self._make_layer(576,[160,128,160,128,160,160,96])
        self.layer7 = self._make_layer(576,[96,128,192,160,192,192,96])
        self.layer8 = nn.Sequential(strideblock(576,[128,192,192,256,256]))
        self.layer9 = self._make_layer(1024,[352,192,320,160,224,224,128])
        self.layer10 = self._make_layer(1024,[352,192,320,192,224,224,128])
        self.avgool = nn.AvgPool2d(7)
        self.fc = nn.Sequential(
        nn.Dropout(p=0.8),
        nn.Linear(1024,num_class),
        )
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] *m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5 )
            elif  isinstance(m,nn.Linear):
                m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()
    def forward(self,x):
        x = self.bottle(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x =self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x =self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x =self.layer9(x)
        x =self.layer10(x)
        x = self.avgool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        pro = F.softmax(x)
        return x , pro

    def _make_layer(self,in_channel,layers):
        blocks = []
        blocks.append(self.block(in_channel,layers))
        return nn.Sequential(*blocks)
"""
InceptionV3  model from http://arxiv.org/abs/1512.00567.
"""

class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,k,s,p,bias=False):
        super(ConvBlock,self).__init__()
        self.conv =nn.Sequential( nn.Conv2d(
        in_channel,out_channel,kernel_size=k,stride=s, padding=p,bias=bias),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
        )
    def forward(self,x):
        x = self.conv(x)
        return x
class InceptionV3base(nn.Module):
    def __init__(self, in_channel,layers=[64,48,64,64,96,96,32] ):
        super(InceptionV3base,self).__init__()
        self.branch1 = nn.Sequential(
        nn.Conv2d(in_channel,layers[0],kernel_size=1,stride=1,padding=0 ,bias=False),
        nn.BatchNorm2d(layers[0]),
        nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
        nn.Conv2d(in_channel,layers[1],1,stride=1,padding=0,bias=False),
        nn.BatchNorm2d(layers[1]),
        nn.ReLU(inplace=True),
        nn.Conv2d(layers[1],layers[2],5,stride=1,padding=2,bias=False),
        nn.BatchNorm2d(layers[2]),
        nn.ReLU(True),
        )
        self.branch3 = nn.Sequential(
        nn.Conv2d(in_channel,layers[3],kernel_size=1,stride=1,padding=0,bias=False),
        nn.BatchNorm2d(layers[3]),
        nn.ReLU(inplace=True),
        nn.Conv2d(layers[3],layers[4],kernel_size=3,stride=1,padding=1,bias=False),
        nn.BatchNorm2d(layers[4]),
        nn.ReLU(inplace=True),
        nn.Conv2d(layers[4],layers[5],kernel_size=3,stride=1,padding=1,bias=False),
        nn.BatchNorm2d(layers[5]),
        nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
        nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
        nn.Conv2d(in_channel,layers[6],1,stride=1,padding=0,bias=False),
        nn.BatchNorm2d(layers[6]),
        nn.ReLU(inplace=True),
        )
    def forward(self,x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1,b2,b3,b4],dim=1)
class InceptionV3_discount(nn.Module):
    def __init__(self,in_channel,layers=[384,64,96,96],block=ConvBlock):
        super(InceptionV3_discount,self).__init__()
        self.branch1 = nn.Sequential(
        block(in_channel,layers[0],3,2,0),
        )
        self.branch2 = nn.Sequential(
        block(in_channel,layers[1],1,1,0),
        block(layers[1],layers[2],3,1,1),
        block(layers[2],layers[3],3,2,0),
        )
        self.branch3 = nn.Sequential(
        nn.MaxPool2d(3,2,padding=0),
        )
    def forward(self,x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1,b2,b3],dim=1)
#
# #     1*7 and 7*1 block
#
class InceptionV3Block(nn.Module):
    def __init__(self,in_channel,layers=[192,128,128,192,128,128,128,128,192,192],block=ConvBlock):
        super(InceptionV3Block,self).__init__()
        self.branch1 = nn.Sequential(
        block(in_channel,layers[0],1,1,0),)
        self.branch2 = nn.Sequential(
        block(in_channel,layers[1],1,1,0),
        block(layers[1],layers[2],(1,7),1,(0,3)),
        block(layers[2],layers[3],(7,1),1,(3,0)),
        )
        self.branch3 = nn.Sequential(
            block(in_channel,layers[4],1,1,0),
            block(layers[4],layers[5],(7,1),1,(3,0)),
            block(layers[5],layers[6],(1,7),1,(0,3)),
            block(layers[6],layers[7],(7,1),1,(3,0)),
            block(layers[7],layers[8],(1,7),1,(0,3)),
        )
        self.branch4 = nn.Sequential(
        nn.AvgPool2d(3,1,padding=1),
        block(in_channel,layers[9],1,1,0),
        )
    def forward(self,x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1,b2,b3,b4],dim=1)
#
#
#    3*3
#
class InceptionV3Substraction(nn.Module):
    def __init__(self,in_channel,layers=[192,320,192,192,192,192],block=ConvBlock):
        super(InceptionV3Substraction,self).__init__()
        self.branch1 = nn.Sequential(
        block(in_channel,layers[0],1,1,0),
        block(layers[0],layers[1],3,2,0),
        )
        self.branch2 = nn.Sequential(
        block(in_channel,layers[2],1,1,0),
        block(layers[2],layers[3],(1,7),1,(0,3)),
        block(layers[3],layers[4],(7,1),1,(3,0)),
        block(layers[4],layers[5],(3,3),2,0),
        )
        self.branch3 = nn.Sequential(
        nn.MaxPool2d(3,stride=2,padding=0),
        )
    def forward(self,x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1,b2,b3],dim=1)
class Concat(nn.Module):
    def __init__(self,in_channel,layers=[384,384],block=ConvBlock):
        super(Concat,self).__init__()
        self.branch1 = nn.Sequential(
        block(in_channel,layers[0],(1,3),1,(0,1)),
        )
        self.branch2 = nn.Sequential(
        block(in_channel,layers[1],(3,1),1,(1,0)),
        )
    def forward(self,x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        return torch.cat([b1,b2],dim=1)

class InceptionLastLayer(nn.Module):
    def __init__(self,in_channel,layers=[320,384,384,448,384,384,384,192],block=ConvBlock,concat=Concat):
        super(InceptionLastLayer,self).__init__()
        self.branch1 = nn.Sequential(
        block(in_channel,layers[0],1,1,0),
        )
        self.branch2 = nn.Sequential(
        block(in_channel,layers[1],1,1,0),
        block(layers[1],layers[2],(1,3),1,(0,1)),
        block(layers[2],layers[3],(3,1),1,(1,0)),
        )
        self.branch3 = nn.Sequential(
        block(in_channel,layers[4],(3,3),1,1),
        concat(layers[4],[layers[5],layers[6]] ),
        )
        self.branch4 = nn.Sequential(
        nn.AvgPool2d(3,1,padding=1),
        block(in_channel,layers[7],1,1,0),)
    def forward(self,x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1,b2,b3,b4],dim=1)


Inceptionv3_shape = 299
class InceptionV3(nn.Module):
    def __init__(self,num_class,block=ConvBlock,base=InceptionV3base,dicount=InceptionV3_discount,base7block=InceptionV3Block,
    Substraction=InceptionV3Substraction,lastblock=InceptionLastLayer):
        super(InceptionV3,self).__init__()
        self.bottle = nn.Sequential(
        block(3,32,3,2,0),
        block(32,32,3,1,0),
        block(32,64,3,1,1),
        nn.MaxPool2d(3,2,padding=0),
        block(64,80,1,1,0),
        block(80,192,3,1,0),
        nn.MaxPool2d(3,2,padding=0),
        )
        ##### 192*35*35
        self.layer1 = nn.Sequential(base(192))
        self.layer2 = nn.Sequential(base(256,[64,48,64,64,96,96,64]))
        self.layer3 = nn.Sequential(base(288,[64,48,64,64,96,96,64]))
        self.layer4 = nn.Sequential(dicount(288))
        self.layer5 = nn.Sequential(base7block(768))
        self.layer6 = nn.Sequential(base7block(768,[192,160,160,192,160,160,160,160,192,192]))
        self.layer7 = nn.Sequential(base7block(768,[192,160,160,192,160,160,160,160,192,192]))
        self.layer8 = nn.Sequential(base7block(768,[192,192,192,192,192,192,192,192,192,192]))
        self.layer9 = nn.Sequential(Substraction(768))
        self.layer10 = nn.Sequential(
        lastblock(1280),)
        self.layer11 = nn.Sequential(lastblock(1728))
        self.avg = nn.AvgPool2d(8,stride=1,padding=0)
        self.fc = nn.Sequential(nn.Dropout(p=0.8),nn.Linear(1728,num_class),)
        self.soft = nn.Softmax(dim=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif  isinstance(m,nn.Linear):
                m.weight.data.normal_(0,0.001)
                m.bias.data.zero_()
    def forward(self,x):
        x = self.bottle(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x =self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x =self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x =self.layer9(x)
        x =self.layer10(x)
        x = self.layer11(x)
        x = self.avg(x)
        x = x.view(x.size(0),-1)
        x =self.fc(x)
        prob = self.soft(x)
        return x , prob
"""
 Inception V4
"""



####
#
#      DenseNet Structure
#
