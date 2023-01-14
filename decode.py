import torch
import pickle as pkl#保存处理好的数据
from my_transformer import Transformer
from pre_process import dataRead, buildInDataset, buildOutDataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # gpu推理/cpu推理

def greedy_decoder(model, enc_input, start_symbol,pad_size):
    """
    贪婪译码方法
    """
    enc_outputs = model.encoder(enc_input)#编码器最后的信息输出
    dec_input = torch.zeros(1, pad_size).type_as(enc_input.data)#0输入当作解码器在测试中的输入
    next_symbol = start_symbol
    model.eval()
    for i in range(0, pad_size):
        dec_input[0][i] = next_symbol
        dec_outputs = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input

def seni2senw1(seni,vocab_i2w):
    sen=[vocab_i2w.get(i.item(),'unk') for i in seni]
    return sen

def seni2senw2(seni,vocab_i2w):
    sen=[vocab_i2w.get(i,'unk') for i in seni]
    return sen

def belu(true_sen,pred_sen):
    pass

if __name__=='__main__':
    in2i_list=pkl.load(open('./save/in2i_list_test.pkl', 'rb'))
    out2i_listB = pkl.load(open('./save/out2i_listB_test.pkl', 'rb'))  # 以二进制保存词典
    out2i_listE = pkl.load(open('./save/out2i_listE_test.pkl', 'rb'))  # 以二进制保存词典
    vocab = pkl.load(open('./save/vocab.pkl', 'rb'))
    #print(vocab)
    vocab_i2w = pkl.load(open('./save/vocab_i2w.pkl', 'rb'))
    #print(in2i_list[555])
    #print(out2i_listB[555])
    model = Transformer(len(vocab),len(vocab)).to(device)
    model.load_state_dict(torch.load('./save_model/transformerepoch500.bin'))

    enc_input=in2i_list[24]
    trg_output=out2i_listB[24]
    print('上联：',seni2senw2(enc_input,vocab_i2w))
    print('下联：',seni2senw2(trg_output,vocab_i2w))
    #print(seni2senw(enc_input,vocabEN))
    enc_input=torch.tensor(enc_input).to(device)
    enc_input=torch.unsqueeze(enc_input,0)
    a=greedy_decoder(model, enc_input, start_symbol=4435,pad_size=32)
    #print(list(a[0]))

    print('机器给出的下联：',seni2senw1(list(a[0]),vocab_i2w))

