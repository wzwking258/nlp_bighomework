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
from train import myDataLoader
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # gpu推理/cpu推理

def test(model,criterion,data_iter):
    loss_sum = 0.0
    sample_num = 0
    model.eval()
    for enc_inputs, dec_inputs,target_batch in data_iter:
        outputs = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, target_batch.contiguous().view(-1)).mean()
        batch_size=enc_inputs.shape[0]
        #print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss_sum += loss.item() * batch_size
        sample_num += batch_size

    loss_mean=loss_sum/sample_num
    ppl=math.exp(loss_mean)
    o_str = 'test_loss: {:6f},test_ppl:{:6f}'.format(loss_mean,ppl)
    print(o_str)

if __name__=='__main__':
    in2i_list_test=pkl.load(open('./save/in2i_list_test.pkl', 'rb'))
    out2i_listB_test = pkl.load(open('./save/out2i_listB_test.pkl', 'rb'))  # 以二进制保存词典
    out2i_listE_test = pkl.load(open('./save/out2i_listE_test.pkl', 'rb'))  # 以二进制保存词典
    vocab = pkl.load(open('./save/vocab.pkl', 'rb'))
    vocab_i2w = pkl.load(open('./save/vocab_i2w.pkl', 'rb'))
    data=list(zip(in2i_list_test,out2i_listB_test,out2i_listE_test))
    data_iter=myDataLoader(batch_size=128,shuffle=True,Data=data)

    model = Transformer(len(vocab), len(vocab)).to(device)
    model.load_state_dict(torch.load('./save_model/transformerepoch500.bin'))

    criterion = nn.CrossEntropyLoss()
    test(model, criterion, data_iter)