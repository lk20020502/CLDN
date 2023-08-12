import torch
import torch.nn as nn
import torch.nn.functional as F

#from modeling.networks import build_feature_extractor, NET_OUT_DIM
from modeling.networks import NET_OUT_DIM, build_feature_extractor
from modeling.networks.resnet18 import complex_feature_renet, simple_feature_renet, feature_renet_wadi, \
    backbone_wadi, backbone_cdd, backbone_msl, backbone_swat, complex_swat_backbone, complex_msl_backbone
from modeling.simple_encoder import feature_extractor

class _SemiADNet_(nn.Module):
    def __init__(self, args):
        super(_SemiADNet_, self).__init__()
        self.args = args
        #self.feature_extractor = build_feature_extractor(self.args.backbone)
        self.feature_extractor=feature_renet_wadi(args)
        self.layer3=nn.Linear(80,50)
        self.layer4=nn.Linear(50,15)
        self.layer5=nn.Linear(15,1)
        self.relu=nn.ReLU()
        #self.conv = nn.Conv2d(in_channels=NET_OUT_DIM[self.args.backbone], out_channels=1, kernel_size=1, padding=0)


    def forward(self, data):

        if self.args.n_scales == 0:
            raise ValueError

        image_pyramid = list()
        for s in range(self.args.n_scales):
            #image_scaled = F.interpolate(data, size=self.args.img_size // (2 ** s)) if s > 0 else data
            #先把照片每个像素点的值降低一些
            feature = self.feature_extractor(data)
            #从图片里面提取特征
            #进行卷积
            scores=[]
            for i in range(len(feature)):
                score=self.layer3(feature[i])
                score=self.relu(score)
                score=self.layer4(score)
                score=self.layer5(score)
                scores.append(score)
            scores=torch.Tensor(scores)




        '''if self.args.topk > 0:
                scores = scores.view(int(scores.size(0)), -1)
                topk = max(int(scores.size(1) * self.args.topk), 1)
                scores = torch.topk(torch.abs(scores), topk, dim=1)[0]
                scores = torch.mean(scores, dim=1).view(-1, 1)
            else:
                scores = scores.view(int(scores.size(0)), -1)
                scores = torch.mean(scores, dim=1).view(-1, 1)

            image_pyramid.append(scores)
        scores = torch.cat(image_pyramid, dim=1)
        score = torch.mean(scores, dim=1)
'''
        return scores.view(-1, 1)









class SemiADNet(nn.Module):
    def __init__(self, args):
        super(SemiADNet, self).__init__()
        self.args = args
        self.feature_extractor = backbone_swat()
        self.conv = nn.Conv2d(in_channels=125, out_channels=1, kernel_size=1, padding=0)
        self.layer1=nn.Linear(27,9)
        self.relu=nn.ReLU()
        self.layer2=nn.Linear(9,1)


    def forward(self, image):

        if self.args.n_scales == 0:
            raise ValueError

        image_pyramid = list()
        for s in range(self.args.n_scales):
            #image_scaled = F.interpolate(image, size=self.args.img_size // (2 ** s)) if s > 0 else image
            feature = self.feature_extractor(image)
            scores = self.conv(feature)
        if self.args.pattern=='topk':
            if self.args.topk > 0:
                scores = scores.view(int(scores.size(0)), -1)
                topk = max(int(scores.size(1) * self.args.topk), 1)
                scores = torch.topk(torch.abs(scores), topk, dim=1)[0]
                scores = torch.mean(scores, dim=1).view(-1, 1)
                image_pyramid.append(scores)
            else:
                scores = scores.view(int(scores.size(0)), -1)
                scores = torch.mean(scores, dim=1).view(-1, 1)

        
                image_pyramid.append(scores)

            scores = torch.cat(image_pyramid, dim=1)
            score = torch.mean(scores, dim=1)
            score=score.view(-1, 1)
        if self.args.pattern == 'linear':
            score=scores.view(1,-1)
            score=self.layer1(score)
            score=self.relu(score)
            score=self.layer2(score)
        return score

    '''
                scores = self.conv(feature).reshape(-1)
                scores=self.relu(scores)
                scores=self.layer1(scores)
                scores =self.relu(scores)
                scores=self.layer2(scores)

           '''
       # return scores


class SemiADNet_2(nn.Module):
    def __init__(self, args):
        super(SemiADNet_2, self).__init__()
        self.args = args
        self.feature_extractor = complex_msl_backbone()
        self.conv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, padding=0)

        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(5,1)

    def forward(self, image):

        if self.args.n_scales == 0:
            raise ValueError

        image_pyramid = list()
        for s in range(self.args.n_scales):
            # image_scaled = F.interpolate(image, size=self.args.img_size // (2 ** s)) if s > 0 else image
            feature = self.feature_extractor(image)

            scores = self.conv(feature)

        if self.args.pattern == 'topk':
            if self.args.topk > 0:
                scores = scores.view(int(scores.size(0)), -1)
                topk = max(int(scores.size(1) * self.args.topk), 1)
                scores = torch.topk(torch.abs(scores), topk, dim=1)[0]
                scores = torch.mean(scores, dim=1).view(-1, 1)
                image_pyramid.append(scores)
            else:
                scores = scores.view(int(scores.size(0)), -1)
                scores = torch.mean(scores, dim=1).view(-1, 1)

                image_pyramid.append(scores)

            scores = torch.cat(image_pyramid, dim=1)
            score = torch.mean(scores, dim=1)
            score = score.view(-1, 1)
        if self.args.pattern == 'linear':
            score = scores.view(1, -1)

            #score = self.layer1(score)
            score = self.relu(score)
            score = self.layer2(score)
        return score

    '''
                scores = self.conv(feature).reshape(-1)
                scores=self.relu(scores)
                scores=self.layer1(scores)
                scores =self.relu(scores)
                scores=self.layer2(scores)

           '''
    # return scores