import numpy as np
import torch
import torch.nn as nn

import argparse
import os
from random import random
from dataloaders.dataloader import build_dataloader, build_dataloader2, build_dataloader_train, build_dataloader_test
from modeling.net import SemiADNet
from tqdm import tqdm

from modeling.networks.resnet18 import backbone_wadi,  simple_feature_renet
from utils import aucPerformance, otherPerformance

from modeling.layers import build_criterion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N=12
gamma=2/3
def curriculum1(s_score,t_score,process):
    x=(s_score+t_score)/10-process  #[-1,1]
    p=1/(1)
    value=random()
    e=2.7183
    p=((1+e)/(1+e**(-x))-1)/(e-1)

    if(p>value):
        output=1
    if(p<value):
        output=0
    return output

class Trainer(object):

    def __init__(self, args):
        self.args = args

        # Define Dataloader
        kwargs = {'num_workers': args.workers}
        self.train_loader=build_dataloader_train(args, **kwargs)
        self.test_loader = build_dataloader_test(args, **kwargs)
        self.teacher=SemiADNet(args)
        self.model =SemiADNet(args)
        self.teacher.load_state_dict(torch.load("C://Users//PC//Desktop//deviation-network-image-main//experiment//swat-cl-experiment-1.pkl"))
        self.model.load_state_dict(torch.load("C://Users//PC//Desktop//deviation-network-image-main//experiment//swat-cl-experiment-2.pkl"))

        self.criterion = build_criterion(args.criterion)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, weight_decay=1e-7)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.3)
        self.optimizer_t = torch.optim.Adam(self.model.parameters(), lr=0.0002, weight_decay=1e-7)
        self.scheduler_t = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.3)
        self.batch_size=args.batch_size
        if args.cuda:
           self.model = self.model.cuda()
           self.criterion = self.criterion.cuda()

    def train(self):
        for epoch in range(0, trainer.args.epochs_for_teacher):
            train_loss_t = 0
            self.tecaher.train()
            tbar = tqdm(self.train_loader)
            for i, sample in enumerate(tbar):
                image, target = sample[0], sample[1]
                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()

                output = self.teacher(image)
                loss = self.criterion(output, target.float())

                self.optimizer_t.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), 1.0)
                self.optimizer_t.step()
                train_loss_t += loss.item()
                tbar.set_description('Teacher network training:Epoch:%d, Train loss: %.3f' % (epoch, train_loss_t / (i + 1)))  # 此处会产生假的epochloss
            self.scheduler_t.step()

        print("The teacher model has been trained... ...")
        scores=[]#存下所有数据点得分
        ano_list=[]#存下所有异常数据点得分
        tbar = tqdm(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample[0], sample[1]
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.teacher(image.float())
            if target == 1:
                ano_list.append(output)
            scores.append(output)
        ano_list.sort()

        n=len(ano_list)//N

        thresholds=[0]
        for i in range(1,N):
            thresholds.append(ano_list[i*n])
        thresholds.append(ano_list[len(ano_list)-1]+1)
                #threshold列表中包括(N-1)个临界异常值
        print(thresholds)
        print("Then student model will be trained... ...")

        stage_length=trainer.args.epochs//(2*N)
        for epoch in range(0, trainer.args.epochs):
            skip = 0
            s=epoch//stage_length#表示处于第几个训练阶段[0,n-1]
            t=epoch-s*stage_length
            T=stage_length
            #if s==0:
             #   threshold=thresholds[N-1]
            if(s>=N):
                threshold=-1
            else:
                threshold=2*(thresholds[N-s]-thresholds[N-1-s])*((t/T)**3)+3*(thresholds[N-s-1]-thresholds[N-s])*((t/T)**2)+thresholds[N-s]
            print(threshold)
            train_loss=0
            self.model.train()
            tbar = tqdm(self.train_loader)
            for i, sample in enumerate(tbar):
                image, target = sample[0], sample[1]
                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()

                output = self.model(image)
                loss = self.criterion(output, target.float())
                self.optimizer.zero_grad()
                if(target==1 and scores[i]<=threshold):
                    skip+=1
                    continue
                if(target==0 and curriculum1(output,scores[i],epoch/self.args.epochs)==1 and (epoch/self.args.epochs)<gamma):
                    skip+=1
                    continue




                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()
                tbar.set_description('Student network training:Epoch:%d, Train loss: %.3f' % (epoch, train_loss / (i + 1 - skip)))#此处会产生假的epochloss
            self.scheduler.step()

    def eval(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        total_pred = np.array([])
        total_target = np.array([])
        pred_target=np.array([])
        for i, sample in enumerate(tbar):
            image, target = sample[0], sample[1]
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image.float())


            if(-1.64<output<1.64):
                pred_target=np.append(pred_target,0)


            else:
                pred_target=np.append(pred_target,1)

            print(output)
            print(target)



            loss = self.criterion(output, target.unsqueeze(1).float())

            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            total_pred = np.append(total_pred, output.data.cpu().numpy())
            total_target = np.append(total_target, target.cpu().numpy())

        roc, pr = aucPerformance(total_pred, total_target)
        print("\n")
        precision, recall, F1=otherPerformance(total_target,pred_target)
        return roc, pr, precision, recall, F1

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), os.path.join(args.experiment_dir, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="batch size used in SGD")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="the number of batches per epoch")
    parser.add_argument("--epochs_for_teacher", type=int, default=0, help="the number of epochs for teacher network")
    parser.add_argument("--epochs", type=int, default=0, help="the number of epochs")
    parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--no_cuda', action='store_true', default=True, help='disables CUDA training')
    parser.add_argument('--weight_name', type=str, default='swat-cl-experiment-2.pkl', help="the name of model weight")
    parser.add_argument('--dataset_root', type=str, default='./data/mvtec_anomaly_detection', help="dataset root")
    parser.add_argument('--experiment_dir', type=str, default='./experiment', help="experiment dir root")
    parser.add_argument('--classname', type=str, default='carpet', help="the subclass of the datasets")
    parser.add_argument('--img_size', type=int, default=448, help="the image size of input")
    parser.add_argument("--n_anomaly", type=int, default=10, help="the number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=1, help="number of scales at which features are extracted")
    parser.add_argument('--backbone', type=str, default='resnet18', help="the backbone network")
    parser.add_argument('--criterion', type=str, default='deviation', help="the loss function")
    parser.add_argument("--topk", type=float, default=0.1, help="the k percentage of instances in the topk module")
    parser.add_argument('--reinforce', type=int, default=2, help="Reinforce the anomaly data")
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    trainer = Trainer(args)
    torch.manual_seed(args.ramdn_seed)

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    argsDict = args.__dict__
    with open(args.experiment_dir + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


    trainer.train()
    trainer.eval()
    trainer.save_weights(args.weight_name)