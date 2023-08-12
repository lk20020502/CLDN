import torch
import torch.nn as nn
from torchvision import models


class feature_resnet18(nn.Module):
    def __init__(self):
        super(feature_resnet18, self).__init__()
        self.net = models.resnet18(pretrained=True)
    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        return x

class feature_resnet50(nn.Module):
    def __init__(self):
        super(feature_resnet50, self).__init__()
        self.net = models.resnet50(pretrained=True)
    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        return x






class Semi_Linear(nn.Module):
    def __init__(self,in_node,out_node,height,in_channel):
        super(Semi_Linear,self).__init__()
        self.layers=nn.Linear(in_node,out_node,height)
        self.height=height
        self.in_channel=in_channel
    def forward(self, x):
        outputs=[]
        for j in range(self.in_channel):
            layers=[]
            for i in range(self.height):
                layers.append(self.layers(x[j][i]))
            y=torch.stack(layers,dim=0)
            outputs.append(y)

        output = torch.stack(outputs, dim=0)
        return output




class backbone_msl(nn.Module):
    def __init__(self):
        super(backbone_msl, self).__init__()
        #
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(3, 1), padding=(1, 0))
        self.layer1=Semi_Linear(27,18,5,5)
        self.conv2=nn.Conv2d(in_channels=5,out_channels=25,kernel_size=(3,1),padding=(1,0))
        self.relu1=nn.ReLU()

        self.conva = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 1), padding=(0, 0))
        self.reneta=Semi_Linear(27,18,5,1)

        self.conv3=nn.Conv2d(in_channels=25,out_channels=75,kernel_size=(2,1))
        self.layer2 = Semi_Linear(18, 9, 4,75)
        self.relu2 = nn.ReLU()
        self.conv4=nn.Conv2d(in_channels=75,out_channels=125,kernel_size=(2,1))
        self.convb=nn.Conv2d(in_channels=25,out_channels=125,kernel_size=(3,1))
        self.renetb=Semi_Linear(18,9,5,25)

    def forward(self, x):
        z = self.conv1(x)
        z=self.layer1(z)
        z = self.relu1(z)
        z = self.conv2(z)
        y =self.reneta(x)
        y=self.conva(y)
        x=z+y
        z = self.conv3(x)
        z=self.layer2(z)
        z = self.relu2(z)
        z = self.conv4(z)
        y=self.renetb(x)
        y = self.convb(y)
        x=z+y
        return x



class feature_renet_wadi(nn.Module):
    def __init__(self):
        super(feature_renet_wadi, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(5, 1), padding=(1, 0))
        self.layer1=Semi_Linear(50,30,8,5)
        self.conv2=nn.Conv2d(in_channels=5,out_channels=25,kernel_size=(5,1),padding=(1,0))
        self.relu1=nn.ReLU()

        self.conva = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(5, 1), padding=(0, 0))
        self.reneta=Semi_Linear(50,30,10,1)

        self.conv3=nn.Conv2d(in_channels=25,out_channels=75,kernel_size=(3,1))
        self.layer2 = Semi_Linear(30, 15, 4,75)
        self.relu2 = nn.ReLU()
        self.conv4=nn.Conv2d(in_channels=75,out_channels=125,kernel_size=(3,1))
        self.convb=nn.Conv2d(in_channels=25,out_channels=125,kernel_size=(5,1))
        self.renetb=Semi_Linear(30,15,6,25)

    def forward(self, x):
        z = self.conv1(x)
        z=self.layer1(z)
        z = self.relu1(z)
        z = self.conv2(z)
        y =self.reneta(x)
        y=self.conva(y)
        x=z+y
        z = self.conv3(x)
        z=self.layer2(z)
        z = self.relu2(z)
        z = self.conv4(z)
        y=self.renetb(x)
        y = self.convb(y)
        x=z+y
        return x



class backbone_swat(nn.Module):
    def __init__(self):
        super(backbone_swat, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(3, 1), padding=(1, 0))
        self.layer1=Semi_Linear(50,30,5,5)
        self.conv2=nn.Conv2d(in_channels=5,out_channels=25,kernel_size=(3,1),padding=(1,0))
        self.relu1=nn.ReLU()

        self.conva = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 1), padding=(0, 0))
        self.reneta=Semi_Linear(50,30,5,1)

        self.conv3=nn.Conv2d(in_channels=25,out_channels=75,kernel_size=(2,1))
        self.layer2 = Semi_Linear(30, 15, 4,75)
        self.relu2 = nn.ReLU()
        self.conv4=nn.Conv2d(in_channels=75,out_channels=125,kernel_size=(2,1))
        self.convb=nn.Conv2d(in_channels=25,out_channels=125,kernel_size=(3,1))
        self.renetb=Semi_Linear(30,15,5,25)

    def forward(self, x):
        z = self.conv1(x)
        z=self.layer1(z)
        z = self.relu1(z)
        z = self.conv2(z)
        y =self.reneta(x)
        y=self.conva(y)
        x=z+y
        z = self.conv3(x)
        z=self.layer2(z)
        z = self.relu2(z)
        z = self.conv4(z)
        y=self.renetb(x)
        y = self.convb(y)
        x=z+y
        return x





class backbone_cdd(nn.Module):
    def __init__(self):
        super(backbone_cdd, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(3, 1), padding=(1, 0))
        self.layer1=Semi_Linear(34,20,5,5)
        self.conv2=nn.Conv2d(in_channels=5,out_channels=25,kernel_size=(3,1),padding=(1,0))
        self.relu1=nn.ReLU()

        self.conva = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 1), padding=(0, 0))
        self.reneta=Semi_Linear(34,20,5,1)

        self.conv3=nn.Conv2d(in_channels=25,out_channels=75,kernel_size=(2,1))
        self.layer2 = Semi_Linear(20, 10, 4,75)
        self.relu2 = nn.ReLU()
        self.conv4=nn.Conv2d(in_channels=75,out_channels=125,kernel_size=(2,1))
        self.convb=nn.Conv2d(in_channels=25,out_channels=125,kernel_size=(3,1))
        self.renetb=Semi_Linear(20,10,5,25)

    def forward(self, x):
        z = self.conv1(x)
        z=self.layer1(z)
        z = self.relu1(z)
        z = self.conv2(z)
        y =self.reneta(x)
        y=self.conva(y)
        x=z+y
        z = self.conv3(x)
        z=self.layer2(z)
        z = self.relu2(z)
        z = self.conv4(z)
        y=self.renetb(x)
        y = self.convb(y)
        x=z+y
        return x









class simple_feature_renet(nn.Module):
    def __init__(self):
        super(simple_feature_renet, self).__init__()
        self.conv0a = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(3, 1), padding=(1, 0))
        self.layer0 = Semi_Linear(127, 81, 5, 5)
        self.conv0b = nn.Conv2d(in_channels=5, out_channels=25, kernel_size=(3, 1), padding=(1, 0))
        self.relu0 = nn.ReLU()

        self.conv0c = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 1), padding=(0, 0))
        self.renet0 = Semi_Linear(127, 81, 5, 1)


        self.conv1 = nn.Conv2d(in_channels=25, out_channels=75, kernel_size=(2, 1))
        self.layer1 = Semi_Linear(81, 30,4 ,75)
        self.conv2 = nn.Conv2d(in_channels=75, out_channels=125, kernel_size=(2, 1))
        self.relu1 = nn.ReLU()

        self.conva = nn.Conv2d(in_channels=25, out_channels=125, kernel_size=(3, 1))
        self.reneta = Semi_Linear(81, 30, 5, 25)

        self.conv0a = nn.Conv2d(in_channels=125, out_channels=25, kernel_size=(2, 1))
        self.layer0 = Semi_Linear(30, 10, 2, 25)
        self.conv0b = nn.Conv2d(in_channels=25, out_channels=1, kernel_size=(2, 1))
        self.relu0 = nn.ReLU()

        self.conv0c = nn.Conv2d(in_channels=125, out_channels=1, kernel_size=(1, 1))
        self.renet0 = Semi_Linear(30, 10, 3, 125)
        self.layer=nn.Linear(10,1)





    def forward(self, x):
        z = self.conv0a(x)
        z = self.layer0(z)
        z = self.relu0(z)
        z = self.conv0b(z)
        y = self.renet0(x)
        y = self.conv0c(y)
        x = z + y
        z = self.conv1(x)
        z = self.layer1(z)
        z = self.relu1(z)
        z = self.conv2(z)
        y = self.reneta(x)
        y = self.conva(y)
        x = z + y
        z = self.conv0a(x)
        z = self.layer0(z)
        z = self.relu0(z)
        z = self.conv0b(z)
        y = self.renet0(x)
        y = self.conv0c(y)
        x = z + y
        x=self.layer(x)

        return x





class complex_msl_backbone(nn.Module):
    def __init__(self):
        super(complex_msl_backbone, self).__init__()
        self.conv0a = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(3, 1), padding=(1, 0))
        self.layer0 = Semi_Linear(27, 18, 5, 5)
        self.conv0b = nn.Conv2d(in_channels=5, out_channels=25, kernel_size=(3, 1), padding=(1, 0))
        self.relu0 = nn.ReLU()

        self.conv0c = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 1), padding=(0, 0))
        self.renet0 = Semi_Linear(27, 18, 5, 1)


        self.conv1 = nn.Conv2d(in_channels=25, out_channels=75, kernel_size=(2, 1))
        self.layer1 = Semi_Linear(18, 10,4 ,75)
        self.conv2 = nn.Conv2d(in_channels=75, out_channels=125, kernel_size=(2, 1))
        self.relu1 = nn.ReLU()

        self.conva = nn.Conv2d(in_channels=25, out_channels=125, kernel_size=(3, 1))
        self.reneta = Semi_Linear(18, 10, 5, 25)

        self.conv0 = nn.Conv2d(in_channels=125, out_channels=300, kernel_size=(2, 1))
        self.layer2 = Semi_Linear(10, 5, 2, 300)
        self.conv0b2 = nn.Conv2d(in_channels=300, out_channels=512, kernel_size=(2, 1))
        self.relu0 = nn.ReLU()

        self.conv0c2 = nn.Conv2d(in_channels=125, out_channels=512, kernel_size=(3, 1))
        self.renet02 = Semi_Linear(10, 5, 3, 125)
        self.layer=nn.Linear(8,1)





    def forward(self, x):
        z = self.conv0a(x)
        z = self.layer0(z)
        z = self.relu0(z)
        z = self.conv0b(z)
        y = self.renet0(x)
        y = self.conv0c(y)
        x = z + y
        z = self.conv1(x)
        z = self.layer1(z)
        z = self.relu1(z)
        z = self.conv2(z)
        y = self.reneta(x)
        y = self.conva(y)
        x = z + y
        z = self.conv0(x)
        z = self.layer2(z)
        z = self.relu0(z)
        z = self.conv0b2(z)
        y = self.renet02(x)
        y = self.conv0c2(y)
        x = z + y
        #x=self.layer(x)

        return x




class complex_swat_backbone(nn.Module):
    def __init__(self):
        super(complex_swat_backbone, self).__init__()
        self.conv0a = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(3, 1), padding=(1, 0))
        self.layer0 = Semi_Linear(50, 35, 5, 5)
        self.conv0b = nn.Conv2d(in_channels=5, out_channels=25, kernel_size=(3, 1), padding=(1, 0))
        self.relu0 = nn.ReLU()

        self.conv0c = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 1), padding=(0, 0))
        self.renet0 = Semi_Linear(50, 35, 5, 1)


        self.conv1 = nn.Conv2d(in_channels=25, out_channels=75, kernel_size=(2, 1))
        self.layer1 = Semi_Linear(35, 20,4 ,75)
        self.conv2 = nn.Conv2d(in_channels=75, out_channels=125, kernel_size=(2, 1))
        self.relu1 = nn.ReLU()

        self.conva = nn.Conv2d(in_channels=25, out_channels=125, kernel_size=(3, 1))
        self.reneta = Semi_Linear(35, 20, 5, 25)

        self.conv0 = nn.Conv2d(in_channels=125, out_channels=300, kernel_size=(2, 1))
        self.layer2 = Semi_Linear(20, 8, 2, 300)
        self.conv0b2 = nn.Conv2d(in_channels=300, out_channels=512, kernel_size=(2, 1))
        self.relu0 = nn.ReLU()

        self.conv0c2 = nn.Conv2d(in_channels=125, out_channels=512, kernel_size=(3, 1))
        self.renet02 = Semi_Linear(20, 8, 3, 125)
        self.layer=nn.Linear(8,1)





    def forward(self, x):
        z = self.conv0a(x)
        z = self.layer0(z)
        z = self.relu0(z)
        z = self.conv0b(z)
        y = self.renet0(x)
        y = self.conv0c(y)
        x = z + y
        z = self.conv1(x)
        z = self.layer1(z)
        z = self.relu1(z)
        z = self.conv2(z)
        y = self.reneta(x)
        y = self.conva(y)
        x = z + y
        z = self.conv0(x)
        z = self.layer2(z)
        z = self.relu0(z)
        z = self.conv0b2(z)
        y = self.renet02(x)
        y = self.conv0c2(y)
        x = z + y
        #x=self.layer(x)

        return x









class backbone_wadi(nn.Module):
    def __init__(self):
        super(backbone_wadi, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(3, 1), padding=(1, 0))
        self.layer1=Semi_Linear(127,75,5,5)
        self.conv2=nn.Conv2d(in_channels=5,out_channels=25,kernel_size=(3,1),padding=(1,0))
        self.relu1=nn.ReLU()

        self.conva = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 1), padding=(0, 0))
        self.reneta=Semi_Linear(127,75,5,1)

        self.conv3=nn.Conv2d(in_channels=25,out_channels=75,kernel_size=(2,1))
        self.layer2 = Semi_Linear(75, 30, 4,75)
        self.relu2 = nn.ReLU()
        self.conv4=nn.Conv2d(in_channels=75,out_channels=125,kernel_size=(2,1))
        self.convb=nn.Conv2d(in_channels=25,out_channels=125,kernel_size=(3,1))
        self.renetb=Semi_Linear(75,30,5,25)

    def forward(self, x):
        z = self.conv1(x)
        z=self.layer1(z)
        z = self.relu1(z)
        z = self.conv2(z)
        y =self.reneta(x)
        y=self.conva(y)
        x=z+y
        z = self.conv3(x)
        z=self.layer2(z)
        z = self.relu2(z)
        z = self.conv4(z)
        y=self.renetb(x)
        y = self.convb(y)
        x=z+y
        return x






class complex_feature_renet(nn.Module):
    def __init__(self):
        super(complex_feature_renet, self).__init__()
        self.conv0a = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(3, 1), padding=(1, 0))
        self.layer0 = Semi_Linear(127, 81, 5, 5)
        self.conv0b = nn.Conv2d(in_channels=5, out_channels=25, kernel_size=(3, 1), padding=(1, 0))
        self.relu0 = nn.ReLU()

        self.conv0c = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 1), padding=(0, 0))
        self.renet0 = Semi_Linear(127, 81, 5, 1)


        self.conv1 = nn.Conv2d(in_channels=25, out_channels=75, kernel_size=(3, 1), padding=(1, 0))
        self.layer1 = Semi_Linear(81, 41, 5, 75)
        self.conv2 = nn.Conv2d(in_channels=75, out_channels=150, kernel_size=(3, 1), padding=(1, 0))
        self.relu1 = nn.ReLU()

        self.conva = nn.Conv2d(in_channels=25, out_channels=150, kernel_size=(1, 1), padding=(0, 0))
        self.reneta = Semi_Linear(81, 41, 5, 25)

        self.conv3 = nn.Conv2d(in_channels=150, out_channels=300, kernel_size=(2, 1))
        self.layer2 = Semi_Linear(41, 18, 4, 300)
        self.relu2 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=300,out_channels=512, kernel_size=(2, 1))
        self.convb = nn.Conv2d(in_channels=150, out_channels=512, kernel_size=(3, 1))
        self.renetb = Semi_Linear(41, 18, 5, 150)

    def forward(self, x):
        z = self.conv0a(x)
        z = self.layer0(z)
        z = self.relu0(z)
        z = self.conv0b(z)
        y = self.renet0(x)
        y = self.conv0c(y)
        x = z + y
        z = self.conv1(x)
        z = self.layer1(z)
        z = self.relu1(z)
        z = self.conv2(z)
        y = self.reneta(x)
        y = self.conva(y)
        x = z + y
        z = self.conv3(x)
        z = self.layer2(z)
        z = self.relu2(z)
        z = self.conv4(z)
        y = self.renetb(x)
        y = self.convb(y)
        x = z + y
        return x