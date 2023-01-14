import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle as pkl#保存处理好的数据

UNK='<unk>'
PAD='<pad>'
BEG='<beg>'
END='<end>'
def dataRead(in_path,out_path):
    '''
    对对联数据集的文本进行读取，保存为列表
    :return:
    '''
    in_list = []#上联
    out_list = []  #下联
    with open(in_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):  # tqdm是 Python 进度条
            if not line or line.count('\n')==len(line):  # 空行：跳过
                continue
            lin = line.strip()
            sen=lin.split(' ')
            in_list.append(sen)
    with open(out_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):  # tqdm是 Python 进度条
            if not line or line.count('\n')==len(line):  # 空行：跳过
                continue
            lin = line.strip()
            sen=lin.split(' ')
            out_list.append(sen)
    return in_list,out_list

def vocabBuild(sen_list,min_freq=0,max_size=None):
    '''
    输入：分好词的句子列表,最小词频，最大字数
    输出：词典
    '''
    vocab_dict = {}
    for sen in tqdm(sen_list):
        for word in sen:
            vocab_dict[word] = vocab_dict.get(word, 0) + 1  # 统计词频
    if max_size!=None:
        vocab_list = sorted([_ for _ in vocab_dict.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]  # 对词频大于等于最小词频的按照词频降序排序
    else:
        vocab_list = sorted([_ for _ in vocab_dict.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)
    vocab_list.insert(0, (PAD, 0))  # 填充需要加在词典的第一个
    vocab_dict = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}  # 构建词典：词-》序号
    vocab_dict.update({BEG: len(vocab_dict),END: len(vocab_dict)+1,UNK:len(vocab_dict)+2})  # 将开始 和结束也加进词典
    return vocab_dict

def buildInDataset(in_list,vocab,pad_size):
    '''
    构建上联的输入
    将所有句子用<pad>填充为统一大小，未知的单词填<unk>.
    '''
    in2i_list=[]
    for sen in tqdm(in_list):
        if len(sen) < pad_size:#填充'<PAD>'
            sen.extend([PAD] * (pad_size - len(sen)))
        else:
            sen = sen[:pad_size]#截断
        in2i=[]
        for word in sen:
            in2i.append(vocab.get(word, vocab.get(UNK)))#对文档生成词编码，未知的设置为unk
        in2i_list.append(in2i)
    return in2i_list

def buildOutDataset(out_list,vocab,pad_size):
    '''
    构建下联的输入
    将所有句子用<pad>填充为统一大小，未知的单词填<unk>.
    decoder输入，以<beg>开头，
    loss的输入，以<end>结尾。
    '''
    out2i_listB=[]#decoder输入，含有beg标记
    out2i_listE = []#loss的输入，含有end标记
    for sen in tqdm(out_list):
        out2i=[]
        senlen=len(sen)
        if len(sen) < pad_size:#填充'<PAD>'
            sen.extend([PAD] * (pad_size - len(sen)-1))
        else:
            sen = sen[:pad_size-1]#截断
        for word in sen:
            out2i.append(vocab.get(word,vocab.get(UNK)))  # 对文档生成词编码
        out2i_listB.append([vocab.get(BEG)]+out2i)
        out2i.insert(senlen, vocab[END])
        out2i_listE.append(out2i)
    return out2i_listB,out2i_listE

if __name__=='__main__':
    in_list,out_list=dataRead('./data/train/in.txt', './data/train/out.txt')
    in_list=in_list[:10000]
    out_list=out_list[:10000]
    #print(len(in_list))
    #print(len(out_list))
    all_list=in_list+out_list
    #print(len(all_list))

    """    len_list = [len(sen) for sen in all_list]
    max_len = max(len_list)# 得到长度最大值:32
    print(max_len)"""

    vocab=vocabBuild(all_list)
    pkl.dump(vocab, open('./save/vocab.pkl', 'wb'))  # pickle模块的序列化操作能够将程序中运行的对象信息保存到文件中去，永久存储
    vocab_i2w={v:k for k,v in vocab.items()}
    pkl.dump(vocab_i2w, open('./save/vocab_i2w.pkl', 'wb'))
    #print(vocab_i2w)

    in2i_list=buildInDataset(in_list, vocab, pad_size=32)
    out2i_listB,out2i_listE=buildOutDataset(out_list,vocab,pad_size=32)
    pkl.dump(in2i_list, open('./save/in2i_list.pkl', 'wb'))
    pkl.dump(out2i_listB, open('./save/out2i_listB.pkl', 'wb'))
    pkl.dump(out2i_listE, open('./save/out2i_listE.pkl', 'wb'))
    print(in2i_list[0])
    print([vocab_i2w[i] for i in in2i_list[0]])
    print(out2i_listB[0])
    print([vocab_i2w[i] for i in out2i_listB[0]])
    print(out2i_listE[0])
    print([vocab_i2w[i] for i in out2i_listE[0]])
    vocab = pkl.load(open('./save/vocab.pkl', 'rb'))
    in_list_test, out_list_test = dataRead('./data/test/in.txt', './data/test/out.txt')

    in2i_list_test = buildInDataset(in_list_test, vocab, pad_size=32)
    out2i_listB_test, out2i_listE_test = buildOutDataset(out_list_test, vocab, pad_size=32)
    pkl.dump(in2i_list_test, open('./save/in2i_list_test.pkl', 'wb'))
    pkl.dump(out2i_listB_test, open('./save/out2i_listB_test.pkl', 'wb'))
    pkl.dump(out2i_listE_test, open('./save/out2i_listE_test.pkl', 'wb'))

