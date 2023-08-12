'''from datasets import mvtecad
from torch.utils.data import DataLoader
from dataloaders.utlis import worker_init_fn_seed, BalancedBatchSampler


def build_dataloader(args, **kwargs):
    train_set = mvtecad.MVTecAD(args, train=True)
    test_set = mvtecad.MVTecAD(args, train=False)
    train_loader = DataLoader(train_set,
                                worker_init_fn=worker_init_fn_seed,
                                batch_sampler=BalancedBatchSampler(args, train_set),
                                **kwargs)
    test_loader = DataLoader(test_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                worker_init_fn=worker_init_fn_seed,
                                **kwargs)
    return train_loader, test_loader
    '''
import argparse

import numpy as np

from datasets import mvtecad
from torch.utils.data import DataLoader
from dataloaders.utlis import worker_init_fn_seed, BalancedBatchSampler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


from torch.utils.data import Dataset, DataLoader
import torch


import xlrd as xd


def build_dataloader2(args, **kwargs):
    workbook1 = xd.open_workbook('C://Users//PC//Desktop//deviation-network-image-main//train+x.xls')
# 通过sheet名称获取
    table1 = workbook1.sheet_by_name(sheet_name='Sheet1')
    workbook2= xd.open_workbook('C://Users//PC//Desktop//deviation-network-image-main//train+y.xls')
# 通过sheet名称获取
    table2 = workbook2.sheet_by_name(sheet_name='Sheet1')
    train_x_list=[]
    train_y_list=[]
    for i in range(table1.nrows):
        table_list = table1.row_values(rowx=i, start_colx=0, end_colx=None)
        train_x_list.append(table_list)
    for i in range(table1.nrows):
        table_list = table2.row_values(rowx=i, start_colx=0, end_colx=None)
        train_y_list.append(table_list)










    class My_dataset(Dataset):
        def __init__(self):
            super(My_dataset,self).__init__()
        # 使用sin函数返回10000个时间序列,如果不自己构造数据，就使用numpy,pandas等读取自己的数据为x即可。
        # 以下数据组织这块既可以放在init方法里，也可以放在getitem方法里
           # self.x = torch.Tensor(train_x_list)
           #self.y = torch.Tensor(train_y_list)

            self.src, self.trg = [], []
            for i in range(len(train_y_list)):
                self.src.append(torch.Tensor(train_x_list[i]))
            for i in range(len(train_y_list)):
                self.trg.append(torch.Tensor(train_y_list[i]))
            self.normal_idx = None
            self.outlier_idx = None

        def __getitem__(self, index):
            return self.src[index], self.trg[index]

        def __len__(self):
            return len(self.src)

        # 或者return len(self.trg), src和trg长度一样


    data_train = My_dataset()
    data_test = My_dataset()
#data_loader_train = DataLoader(data_train, batch_size=5, shuffle=False)
#data_loader_test = DataLoader(data_test, batch_size=5, shuffle=False)

    train_loader = DataLoader(data_train,
                          #worker_init_fn=worker_init_fn_seed,
                          batch_size=args.batch_size,
                          #=BalancedBatchSampler(args, data_train),
                          shuffle=True,
                          drop_last=True)
    test_loader = DataLoader(data_test,
                         batch_size=args.batch_size,
                         shuffle=True,
                         #**kwargs
                         drop_last=True)
    '''
    train_loader = DataLoader(data_train, batch_size=5, shuffle=False)
    test_loader = DataLoader(data_test, batch_size=5, shuffle=False)
'''
# i_batch的多少根据batch size和def __len__(self)返回的长度确定
# batch_data返回的值根据def __getitem__(self, index)来确定
# 对训练集：(不太清楚enumerate返回什么的时候就多print试试)
#    for i_batch, batch_data in enumerate(train_loader):
#        print(i_batch)  # 打印batch编号
#        print(batch_data[0].size())  # 打印该batch里面src
#        print(batch_data[1].size())

    return train_loader, test_loader
































def build_dataloader2(args, **kwargs):
    workbook1 = xd.open_workbook('C://Users//PC//Desktop//deviation-network-image-main//train++x.xls')
# 通过sheet名称获取
    table1 = workbook1.sheet_by_name(sheet_name='Sheet1')
    workbook2= xd.open_workbook('C://Users//PC//Desktop//deviation-network-image-main//train++y.xls')
# 通过sheet名称获取
    table2 = workbook2.sheet_by_name(sheet_name='Sheet1')
    train_x_list=[]
    train_y_list=[]
    for i in range(table1.nrows):
        table_list = table1.row_values(rowx=i, start_colx=0, end_colx=None)
        train_x_list.append(table_list)
    for i in range(table1.nrows):
        table_list = table2.row_values(rowx=i, start_colx=0, end_colx=None)
        train_y_list.append(table_list)
    print(train_x_list)
    print(train_y_list)


    class My_dataset(Dataset):
        def __init__(self):
            super(My_dataset,self).__init__()
        # 使用sin函数返回10000个时间序列,如果不自己构造数据，就使用numpy,pandas等读取自己的数据为x即可。
        # 以下数据组织这块既可以放在init方法里，也可以放在getitem方法里
           # self.x = torch.Tensor(train_x_list)
           #self.y = torch.Tensor(train_y_list)

            self.src, self.trg = [], []
            self.x,self.y=[],[]
            for i in range(5, len(train_y_list)):
                if (train_y_list[i - 4] == 1 and train_y_list[i] == 0):
                    continue
                for j in range(i - 4, i + 1):
                    self.x.append(np.array(train_x_list[j]))

                self.src.append(torch.Tensor(self.x))
                self.x = []
                self.trg.append(torch.Tensor(train_y_list[i]).unsqueeze(0))
            self.normal_idx = None
            self.outlier_idx = None

        def __getitem__(self, index):
            return self.src[index], self.trg[index]

        def __len__(self):
            return len(self.src)

        # 或者return len(self.trg), src和trg长度一样


    data_train = My_dataset()
    data_test = My_dataset()
#data_loader_train = DataLoader(data_train, batch_size=5, shuffle=False)
#data_loader_test = DataLoader(data_test, batch_size=5, shuffle=False)

    train_loader = DataLoader(data_train,
                          #worker_init_fn=worker_init_fn_seed,
                          batch_size=args.batch_size,
                          #=BalancedBatchSampler(args, data_train),
                          shuffle=False,
                          drop_last=True)
    test_loader = DataLoader(data_test,
                         batch_size=args.batch_size,
                         shuffle=False,
                         #**kwargs
                         drop_last=True)
    '''
    train_loader = DataLoader(data_train, batch_size=5, shuffle=False)
    test_loader = DataLoader(data_test, batch_size=5, shuffle=False)
'''
# i_batch的多少根据batch size和def __len__(self)返回的长度确定
# batch_data返回的值根据def __getitem__(self, index)来确定
# 对训练集：(不太清楚enumerate返回什么的时候就多print试试)
#    for i_batch, batch_data in enumerate(train_loader):
#        print(i_batch)  # 打印batch编号
#        print(batch_data[0].size())  # 打印该batch里面src
#        print(batch_data[1].size())

    return train_loader, test_loader







































def build_dataloader(args, **kwargs):
    train_set = mvtecad.MVTecAD(args, train=True)
    test_set = mvtecad.MVTecAD(args, train=False)
    train_loader = DataLoader(train_set,
                                worker_init_fn=worker_init_fn_seed,
                                batch_sampler=BalancedBatchSampler(args, train_set),
                                **kwargs)
    test_loader = DataLoader(test_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                worker_init_fn=worker_init_fn_seed,
                                **kwargs)
    return train_loader, test_loader



















def build_dataloader_train (args, **kwargs):
    workbook1 = xd.open_workbook('C://Users//PC//Desktop//deviation-network-image-main//swat_train_x_0.xls')
# 通过sheet名称获取
    table1 = workbook1.sheet_by_name(sheet_name='Sheet1')
    workbook2= xd.open_workbook('C://Users//PC//Desktop//deviation-network-image-main//swat_train_y_0.xls')
# 通过sheet名称获取
    table2 = workbook2.sheet_by_name(sheet_name='Sheet1')
    train_x_list=[]
    train_y_list=[]
    for i in range(table1.nrows):
        table_list = table1.row_values(rowx=i, start_colx=0, end_colx=None)
        train_x_list.append(table_list)
    for i in range(table1.nrows):
        table_list = table2.row_values(rowx=i, start_colx=0, end_colx=None)
        train_y_list.append(table_list)










    class My_dataset1(Dataset):
        def __init__(self):
            super(My_dataset1,self).__init__()
        # 使用sin函数返回10000个时间序列,如果不自己构造数据，就使用numpy,pandas等读取自己的数据为x即可。
        # 以下数据组织这块既可以放在init方法里，也可以放在getitem方法里
           # self.x = torch.Tensor(train_x_list)
           #self.y = torch.Tensor(train_y_list)

            self.src, self.trg = [], []
            self.x,self.y=[],[]
            for i in range(5, len(train_y_list)):
                if ((train_y_list[i-4][0]==1 or train_y_list[i-2][0]==1) and train_y_list[i][0] == 0):
                    continue
                for j in range(i - 4, i + 1):
                    self.x.append(np.array(train_x_list[j]))
                self.src.append(torch.Tensor(self.x))
                self.x = []
                self.trg.append(torch.Tensor(train_y_list[i]).unsqueeze(0))

            #两个＃之间是用于数据增强的部分
                if(train_y_list[i][0]==1):

                    for t in range(0):
                        for j in range(i - 4, i + 1):
                            self.x.append(np.array(train_x_list[j]))
                        self.src.append(torch.Tensor(self.x))
                        self.x = []
                        self.trg.append(torch.Tensor(train_y_list[i]).unsqueeze(0))

            #
            self.normal_idx = None
            self.outlier_idx = None
        def __getitem__(self, index):
            return self.src[index], self.trg[index]

        def __len__(self):
            return len(self.src)

        # 或者return len(self.trg), src和trg长度一样



        # 或者return len(self.trg), src和trg长度一样


    data_train = My_dataset1()

#data_loader_train = DataLoader(data_train, batch_size=5, shuffle=False)
#data_loader_test = DataLoader(data_test, batch_size=5, shuffle=False)

    train_loader = DataLoader(data_train,
                          #worker_init_fn=worker_init_fn_seed,
                          batch_size=args.batch_size,
                          #=BalancedBatchSampler(args, data_train),
                          shuffle=True,
                          drop_last=True)

    '''
    train_loader = DataLoader(data_train, batch_size=5, shuffle=False)
    test_loader = DataLoader(data_test, batch_size=5, shuffle=False)
'''
# i_batch的多少根据batch size和def __len__(self)返回的长度确定
# batch_data返回的值根据def __getitem__(self, index)来确定
# 对训练集：(不太清楚enumerate返回什么的时候就多print试试)
#    for i_batch, batch_data in enumerate(train_loader):
#        print(i_batch)  # 打印batch编号
#        print(batch_data[0].size())  # 打印该batch里面src
#        print(batch_data[1].size())


    return train_loader



def build_dataloader_test(args, **kwargs):


    workbook3 = xd.open_workbook('C://Users//PC//Desktop//deviation-network-image-main//swat_test_x_1.xls')
        # 通过sheet名称获取
    table3 = workbook3.sheet_by_name(sheet_name='Sheet1')
    workbook4 = xd.open_workbook('C://Users//PC//Desktop//deviation-network-image-main//swat_test_y_1.xls')
        # 通过sheet名称获取
    table4 = workbook4.sheet_by_name(sheet_name='Sheet1')
    train_x_list = []
    train_y_list = []
    for i in range(table4.nrows):
        table_list = table3.row_values(rowx=i, start_colx=0, end_colx=None)
        train_x_list.append(table_list)
    for i in range(table4.nrows):
        table_list = table4.row_values(rowx=i, start_colx=0, end_colx=None)
        train_y_list.append(table_list)


    class My_dataset2(Dataset):
        def __init__(self):
            super(My_dataset2,self).__init__()
        # 使用sin函数返回10000个时间序列,如果不自己构造数据，就使用numpy,pandas等读取自己的数据为x即可。
        # 以下数据组织这块既可以放在init方法里，也可以放在getitem方法里
           # self.x = torch.Tensor(train_x_list)
           #self.y = torch.Tensor(train_y_list)

            self.src, self.trg = [], []
            self.x,self.y=[],[]
            for i in range(5,len(train_y_list)):
                if((train_y_list[i-4][0]==1 or train_y_list[i-2][0]==1) and train_y_list[i][0]==0):
                    continue
                for j in  range(i-4,i+1):
                    self.x.append(np.array(train_x_list[j]))
                self.src.append(torch.Tensor(self.x))
                self.x=[]
                self.trg.append(torch.Tensor(train_y_list[i]).unsqueeze(0))

            self.normal_idx = None
            self.outlier_idx = None

        def __getitem__(self, index):
            return self.src[index], self.trg[index]

        def __len__(self):
            return len(self.src)

        # 或者return len(self.trg), src和trg长度一样



    data_test = My_dataset2()
#data_loader_train = DataLoader(data_train, batch_size=5, shuffle=False)
#data_loader_test = DataLoader(data_test, batch_size=5, shuffle=False)


    test_loader = DataLoader(data_test,
                         batch_size=1,
                         shuffle=True,
                         #**kwargs
                         drop_last=True)
    '''
    train_loader = DataLoader(data_train, batch_size=5, shuffle=False)
    test_loader = DataLoader(data_test, batch_size=5, shuffle=False)
'''
# i_batch的多少根据batch size和def __len__(self)返回的长度确定
# batch_data返回的值根据def __getitem__(self, index)来确定
# 对训练集：(不太清楚enumerate返回什么的时候就多print试试)
#    for i_batch, batch_data in enumerate(train_loader):
#        print(i_batch)  # 打印batch编号
#        print(batch_data[0].size())  # 打印该batch里面src
#        print(batch_data[1].size())


    return test_loader
