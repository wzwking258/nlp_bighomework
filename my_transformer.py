import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # gpu推理/cpu推理
'''超参数设置'''
d_model = 512  # 词嵌入维度
d_ff = 2048  # 最后的全连接层维度
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # 有几个编码层或解码层
n_heads = 8  # 有几个多头注意力

def make_batch():
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)

def get_attn_pad_mask1(enc_inputs):#输入形状都是(batchsize,src_len)
    '''在编码器中对自注意力矩阵生成符号矩阵，屏蔽<pad>对自注意力的影响（设为负无穷）
    一个序列输入形成一个符号矩阵,形状(src_len,src_len)，batchsize个输入需要batchsize个符号矩阵
    输入形状(batchsize,src_len)
    输出形状（batch_size,src_len，src_len）'''
    batch_size,src_len=enc_inputs.size()
    pad_attn_mask=enc_inputs.data.eq(0).unsqueeze(1)#因为词表中0代表pad。是0的返回true，输出bool矩阵，形状经过扩展变成（batch_size,1，src_len）
    pad_attn_mask=pad_attn_mask.expand(batch_size,src_len,src_len)#输出0，1矩阵
    return pad_attn_mask#因为每一批输入都需要一个pad符号矩阵

def get_attn_pad_mask2(enc_inputs,dec_inputs):
    '''对自注意力矩阵生成符号矩阵，屏蔽<pad>对自注意力的影响（设为负无穷）
    因为自注意力矩阵是QKT相乘得到的，形状(tgt_len,src_len)由K，Q矩阵共同决定。所以都要输入
    输入形状分别是enc_inputs（batchsize,src_len）,dec_inputs(batchsize,tgt_len)
    输出形状(batch_size, len_d, len_e)
    '''
    batch_size,len_d=dec_inputs.size()
    batch_size, len_e = enc_inputs.size()
    pad_attn_mask=enc_inputs.data.eq(0).unsqueeze(1)#因为词表中0代表pad。是0的返回true，输出bool矩阵，形状经过扩展变成（batch_size,1，len_e）
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_d, len_e) # 输出0，1矩阵
    return pad_attn_mask

def get_attn_subsequent_mask(dec_inputs):
    '''对输入右边单词进行掩盖的mask矩阵(实际上就是个对角线位置上移一格上三角矩阵)
    输入：dec_inputs[batch_size , target_len]
    输出：subsequent_mask[batch_size , target_len,target_len]
    '''
    attn_shape = [dec_inputs.size(0), dec_inputs.size(1), dec_inputs.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)#生成上三角矩阵，上面全是1，下面全是0.k=1表示对角线位置上移一格
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()#输出0，1矩阵
    return subsequent_mask

class ScaledDotProductAttention(nn.Module):
    '''计算自注意力函数
    输入Q(batch_size,n_heads,sen_len,d_k),K(batch_size,n_heads,sen_len,d_k),V(batch_size,n_heads,sen_len,d_v),
    输入attn_mask[batch_size ,n_heads , sen_len , sen_len]
    输出(batch_size,n_heads,sen_len,d_v)，
    '''
    def __init__(self,dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V, attn_mask):#K形状(batch_size,n_heads,sen_len,d_k)，k的转置是只交换最后2个维度
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size , n_heads , len_q(=len_k) , len_k(=len_q)]
        #attn: [batch_size ,n_heads , len_q(=len_k) , len_k(=len_q)]

        scores.masked_fill_(attn_mask.bool(), -1e9) # 对attn_mask中值为True的位置（pad位置）设置为负无穷。score需要和attn_mask形状相同
        attn = nn.Softmax(dim=-1)(scores)#dim=-1.对每一行求softmax,即每一行和=1。形状不变
        context = torch.matmul(attn, V)#自注意力函数=各个位置的加权attn
        #V形状(batch_size,n_heads,sen_len,d_v)，
        # 相乘后形状(batch_size,n_heads,len_q,d_v)，
        context=self.dropout(context)
        return context

class MultiHeadAttention(nn.Module):
    '''多头注意力层
    这里的W_Q（或W_V 或W_K）把多个头合并成了一个矩阵，形状输出d_k*n_heads
    输入形状：X(batch_size,sen_len,d_model)，attn_mask(batchsize,tgt_len，src_len)
    输出形状output[batch_size ,sen_len , d_model]和X一样
    '''
    def __init__(self):
        super(MultiHeadAttention,self).__init__()
        self.W_Q = nn.Linear(d_model,d_k*n_heads)#在编码层，d_q=d_k,输入形状(b,*,d_model),输出形状(b,*,d_k*n_heads),其中*表示任意维度
        self.W_K = nn.Linear(d_model,d_k*n_heads)
        self.W_V = nn.Linear(d_model,d_v*n_heads)

        self.linear=nn.Linear(d_v*n_heads,d_model)
        self.layer_norm=nn.LayerNorm(d_model)#层归一化，对最后一维进行归一化
    def forward(self,Q,K,V,attn_mask):#输入Q,K,V,形状(batch_size,sen_len,d_model)符号矩阵形状(batchsize,src_len，src_len)
        residual=Q#直接映射
        batch_size=Q.shape[0]
        #输出形状(batch_size,sen_len,d_k*n_heads),需要转换为(batch_size,n_heads,sen_len,d_k)
        Q = self.W_Q(Q).view(batch_size, -1, n_heads,d_k).transpose(1,2)#Q(batch_size,n_heads,sen_len,d_k)
        K = self.W_K(K).view(batch_size, -1, n_heads,d_k).transpose(1,2)#K(batch_size,n_heads,sen_len,d_k)
        V = self.W_V(V).view(batch_size, -1, n_heads,d_v).transpose(1,2)#V(batch_size,n_heads,sen_len,d_v)

        attn_mask=attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # 扩充为和score一样大小
        # context: [batch_size x n_heads x len_q x d_v],

        context = ScaledDotProductAttention()(Q, K, V, attn_mask)#计算自注意力函数，context形状(batch_size,n_heads,len_q,d_v)，
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        # 为了进入全连接层，需要改变context最后一维的形状: [batch_size , * , n_heads * d_v]
        output = self.linear(context)#output形状和输入的x形状相同，(b,*,d_modle)
        return self.layer_norm(output + residual) # output: [batch_size , len_q , d_model]

class PositionwiseFeedForward(nn.Module):
    """实现FFN.一个两层的前馈神经网络。
    输入形状inputs(batch_size ,sen_len , d_model)
    输出形状outputs(batch_size ,sen_len , d_model)
    """
    def __init__(self, d_model, d_ff,dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        residual = inputs
        output=F.relu(self.w_1(inputs))
        output = self.w_2(output)
        return self.dropout(self.layer_norm(output + residual))

class EncoderLayer(nn.Module):
    '''编码层，共n_layer个，包含两个部分，多头注意力机制和Feed Forward层
    输入x(batch_size,sen_len,d_model)，pad_mask（batch_size,src_len，src_len）
    输出enc_outputs(batch_size ,src_len , d_model)
    '''
    def __init__(self):
        super(EncoderLayer,self).__init__()
        self.enc_self_attn=MultiHeadAttention()#多头注意力
        self.pos_ffn=PositionwiseFeedForward(d_model, d_ff)#一个两层的前馈网络,输出和输入形状相同。
    def forward(self,x,pad_mask):
        enc_outputs = self.enc_self_attn(x,x,x,pad_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs

class DecoderLayer(nn.Module):
    '''解码层，共n_layer个，包含两个部分，2个多头注意力机制和Feed Forward层
    输入enc_outputs(batch_size,src_len,d_model)，dec_inputs(batch_size,tgt_len,d_model)
    pad_mask（batch_size,tgt_len，src_len）
    输出enc_outputs(batch_size ,sen_len , d_model)
    '''
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()#第一个多头注意力层和编码层中的相同，除了增加了mask位置t+1的单词的操作
        self.dec_enc_attn = MultiHeadAttention()#第而个多头注意力层使用上一层的Z计算Q，使用编码器的C矩阵计算K，V
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)#X，X，X
        dec_outputs = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)#计算Q的输入，计算K的输入，计算V的输入
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs

class PositonalEncoding(nn.Module):
    '''生成位置编码pe，使得transformer可以利用单词的位置信息，同时也实现了x矩阵和pe矩阵相加的功能
    使用max_len使得句子长度不能超出位置编码的范围，但是如果训练集里面短句子训练数据比较多的话，则对
    测试中的长句子不友好。 Vaswani也在文中提及最后使用正弦位置编码是因为这个原因，适应不同长度的编码。
    所以max_len要远大于src_len和tgt_len。
    输入：x(batch_size，max_len,d_model)
    输出：x+pe(batch_size，max_len,d_model)
    '''
    def __init__(self,d_model,max_len=1000,dropout=0.1):#因为编码器和解码器的最大句长可能不同，所以作为参数max_len输入
        super(PositonalEncoding,self).__init__()

        self.dropout=nn.Dropout(p=dropout)#丢弃法，避免过拟合，如p=0.3表示输入的神经元有p = 0.3的概率激活
        pe=torch.zeros((max_len,d_model))#PE矩阵形状(max_len,d_model)
        pos=torch.arange(0,max_len,dtype=torch.float32).reshape(-1,1)#位置向量，0到max_len,为了广播变形为行向量
        div_term=torch.exp(torch.arange(0,d_model,2,dtype=torch.float32)*(-math.log(10000.0))/d_model).reshape(1,-1)#分母部分，两个公式是一样的。为了广播变形为列向量
        pe[:,0::2]=torch.sin(pos*div_term)#0::2表示从0开始，步长为2.这里利用了广播，pos和div_term都有一维是1，可以广播。广播后形状((max_len,d_model//2))
        pe[:, 1::2] = torch.cos(pos * div_term)

        pe= pe.unsqueeze(0)#PE矩阵形状(1，max_len,d_model)，为了与批量的x相加而增加批度维
        self.register_buffer('pe',pe)#定义一个缓冲区，这个参数不更新???

    def forward(self,x):#x的形状(batch_size，max_len,,d_model),
        x=x+Variable(self.pe[0,:x.shape[1],:],requires_grad=False)
        #x=x+self.pe[:x.size(0),:]???
        return self.dropout(x)


class Encoder(nn.Module):
    '''编码器部分
    输入enc_inputs(batchsize,src_len)，
    输出enc_outputs(batch_size ,src_len , d_model)
    '''
    def __init__(self,src_vocab_size):
        super(Encoder,self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)  # 词嵌入矩阵
        self.pos_emb = PositonalEncoding(d_model) # 位置向量矩阵，因为x=x+p相加，输出形状需要与词嵌入矩阵输出一致
        self.layers=nn.ModuleList([EncoderLayer() for _ in range(n_layers)])#按照顺序添加编码层，共n_layers层

    def forward(self,enc_inputs):
        #encoder输入形状是(batchsize,src_len)
        enc_wordEmbs=self.src_emb(enc_inputs)#输出形状(batchsize,src_len,d_model)
        enc_outputs = self.pos_emb(enc_wordEmbs)#计算位置矩阵并相加，得(batch_size，src_len,d_model)

        #get_attn_pad_mask是为了得到句子中pad的位置信息，为了后面将pad设置为负无穷
        enc_self_attn_mask=get_attn_pad_mask1(enc_inputs)

        for layer in self.layers:#前面一层编码层的输出是后面一层编码层的输入
            enc_outputs=layer(enc_outputs,enc_self_attn_mask)
        return enc_outputs

class Decoder(nn.Module):
    """解码器：包含两个多头注意力层，第一层采用了masked操作，第二层的K，V矩阵使用编码器的最后输出C进行计算，Q使用上一层输出进行计算。
    输入形状：dec_inputs [batch_size , target_len],enc_inputs[batch_size , src_len],enc_outputs[batch_size , src_len,d_model]
    输出形状dec_outputs[batch_size , tgt_len,d_model]
    """
    def __init__(self,tgt_vocab_size):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)#词嵌入矩阵
        self.pos_emb = PositonalEncoding(d_model) # 位置向量矩阵，因为x=x+p相加，输出形状需要与词嵌入矩阵输出一致
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])#按照顺序添加解码层，共n_layers层

    def forward(self, dec_inputs, enc_inputs, enc_outputs): # 输入dec_inputs : [batch_size x target_len]
        dec_outputs = self.tgt_emb(dec_inputs)# 输出dec_outputs[batch_size , tgt_len,d_model]
        dec_outputs = self.pos_emb(dec_outputs)# 输出dec_outputs[batch_size , tgt_len,d_model]
        #获取自注意力矩阵的遮蔽pad的符号矩阵
        dec_self_attn_pad_mask = get_attn_pad_mask1(dec_inputs).to(device)#输出形状（batch_size,tgt_len，tgt_len）
        #获取自注意力矩阵的遮蔽t+1位置的符号矩阵
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs).to(device)#输出形状（batch_size,tgt_len，tgt_len）
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)#把两个矩阵合并为1个矩阵
        #dec_inputs形状（batch_size,tgt_len），enc_inputs形状（batch_size,src_len）
        dec_enc_attn_mask = get_attn_pad_mask2(enc_inputs,dec_inputs).to(device)#形状（batch_size,tgt_len，src_len）
        #dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:#上一层的输入是下一层的输出
            dec_outputs = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
        return dec_outputs

class Transformer(nn.Module):
    '''   可将transformer可以分为3个部分：编码器，解码器，输出层。
    transformer 有两个输入，编码器的输入是待翻译的句子，解码器的输入是翻译完成的句子
    参数设置
    src_vocab_size = 源语言词表大小
    tgt_vocab_size = 目标语言词表大小
    src_len =   源语言句子最大长度，小于填充pad，大于截断
    tgt_len =   目标语言句子最大长度
    输出输出（batch_size*tgt_len,tgt_vocab_size），每一行最大的一个值是最可能单词的概率。
    '''
    def __init__(self,src_vocab_size,tgt_vocab_size):
        super(Transformer,self).__init__()
        self.encoder = Encoder(src_vocab_size)#编码器
        self.decoder = Decoder(tgt_vocab_size)#输出（batch_size,tgt_len,d_model）
        self.projection = nn.Linear(d_model, tgt_vocab_size)#解码器输出的形状与输入的形状一致，输出为词表上每个单词概率的分布。
        ##输出（batch_size,tgt_len,tgt_vocab_size）
    def forward(self,enc_inputs,dec_inputs):
        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits : [batch_size , src_vocab_size , tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1))#输出（batch_size*tgt_len,tgt_vocab_size）

def greedy_decoder(model, enc_input, start_symbol):
    enc_outputs = model.encoder(enc_input)
    dec_input = torch.zeros(1, 5).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, 5):
        dec_input[0][i] = next_symbol
        dec_outputs = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input

if __name__ == '__main__':
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
    # Transformer Parameters
    # Padding Should be Zero index
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5 # length of source
    tgt_len = 5 # length of target

    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention

    model = Transformer(src_vocab_size,tgt_vocab_size).to(device)

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    enc_inputs, dec_inputs, target_batch = make_batch()
    enc_inputs, dec_inputs, target_batch=enc_inputs.to(device), dec_inputs.to(device), target_batch.to(device)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Test
    greedy_dec_input = greedy_decoder(model, enc_inputs, start_symbol=tgt_vocab["S"])
    predict = model(enc_inputs, greedy_dec_input)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])


