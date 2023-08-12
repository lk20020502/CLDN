
import torch
import torch.nn as nn
from torchvision import models


class feature_extractor(nn.Module):
    def __init__(self,args):
        super(feature_extractor, self).__init__()
        self.layer1=nn.Linear(135,110)
        self.layer2=nn.Linear(110,80)
        self.relu=nn.ReLU()
        self.args=args
    def forward(self, x):
        features=[]
        for i in range(self.args.batch_size):
            data=self.layer1(x[i])
            data=self.relu(data)
            data=self.layer2(data)
            features.append(data)

        return features




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