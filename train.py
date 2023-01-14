import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pickle as pkl#保存处理好的数据

from my_transformer import Transformer
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # gpu推理/cpu推理

class myDataset(Dataset):#自定义 Dataset 类必须实现三个函数: init、 len 和 getitem
    """__init__
    输入data=list(zip([in2i_list,out2i_listB,out2i_listE]))，
    分别表示encoder的输入，decoder的输入，计算loss函数的输入
    """
    def __init__(self,data):
        random.shuffle(data)
        self.data = data
        self.device= device

    def __len__(self):#获取样本的个数
        return len(self.data)

    def __getitem__(self, idx):#给定索引 idx ，从数据集中加载并返回该索引的标签和其样本。
        thedata = self.data[idx]
        enc_inputs = torch.tensor(thedata[0]).to(self.device)
        dec_inputs = torch.tensor(thedata[1]).to(self.device)
        target_inputs = torch.tensor(thedata[2]).to(self.device)
        return enc_inputs, dec_inputs, target_inputs

def myDataLoader(batch_size,shuffle,Data):#数据加载迭代器
    dataset = myDataset(Data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)#DataLoader 是一个迭代器

def train(model,optimizer,criterion,data_iter):
    for epoch in range(500):
        loss_sum = 0.0
        sample_num = 0
        model.train()
        time_start = time.time()
        for enc_inputs, dec_inputs,target_batch in data_iter:
             # 记录开始时间

            optimizer.zero_grad()
            outputs = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, target_batch.contiguous().view(-1)).mean()
            batch_size=enc_inputs.shape[0]
            #print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * batch_size
            sample_num += batch_size

        time_end = time.time()  # 记录结束时间
        time_cost = time_end - time_start  # 计算执行时间，单位为秒/s
        loss_mean=loss_sum/sample_num
        ppl=math.exp(loss_mean)
        o_str = 'epoch: {}, loss: {:6f},ppl:{:6f},costtime: {:3f}'.format(epoch+1, loss_mean,ppl,time_cost)
        print(o_str)
        if (epoch+1)%10==0:
            torch.save(model.state_dict(), f'./save_model/transformer{"epoch"+str(epoch+1)}.bin')

if __name__=='__main__':
    in2i_list=pkl.load(open('./save/in2i_list.pkl', 'rb'))
    out2i_listB = pkl.load(open('./save/out2i_listB.pkl', 'rb'))  # 以二进制保存词典
    out2i_listE = pkl.load(open('./save/out2i_listE.pkl', 'rb'))  # 以二进制保存词典
    vocab = pkl.load(open('./save/vocab.pkl', 'rb'))
    vocab_i2w = pkl.load(open('./save/vocab_i2w.pkl', 'rb'))
    data=list(zip(in2i_list,out2i_listB,out2i_listE))
    data_iter=myDataLoader(batch_size=128,shuffle=True,Data=data)

    model = Transformer(len(vocab),len(vocab)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99) #用adam的话效果不好?!
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train(model, optimizer, criterion, data_iter)
    #model.load_state_dict(torch.load('./save_model/transformer.bin'))
