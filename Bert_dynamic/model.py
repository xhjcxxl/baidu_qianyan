import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
import torch.nn.functional as F
import copy


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='bert.embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='bert.embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  # 1024*1024
        self.weight.data.normal_(mean=0.0, std=0.05)  # 初始化

        self.bias = nn.Parameter(torch.Tensor(hidden_size))  # 1024*1

        b = np.zeros(hidden_size, dtype=np.float32)  # 初始设定
        self.bias.data.copy_(torch.from_numpy(b))  # 初始化

        self.query = nn.Parameter(torch.Tensor(hidden_size))  # 应该看作 1024*1
        self.query.data.normal_(mean=0.0, std=0.05)  # 初始化

    def forward(self, batch_hidden, batch_masks):
        # batch_hidden: batch x length x hidden_size (2 * hidden_size of lstm) 这里正好就是BiLSTM模型的输出结果
        # batch_masks:  batch x length 这个就是数据的输入mask

        # key是encoder的各个隐藏状态，就是LSTM的隐藏状态结果
        key = torch.matmul(batch_hidden, self.weight) + self.bias  # batch x length x hidden 8*512*1024

        # compute attention (Q,K)结果就是score得分
        outputs = torch.matmul(key, self.query)  # batch x length 8*512
        # 把output中的对应位置的数字mask为1
        masked_outputs = outputs.masked_fill((1 - batch_masks).bool(), float(-1e32))  # 8*512

        # 进行softmax
        attn_scores = F.softmax(masked_outputs, dim=1)  # batch x length 进行softmax 8*512

        # 对于全零向量，-1e32的结果为 1/len, -inf为nan, 额外补0
        masked_attn_scores = attn_scores.masked_fill((1 - batch_masks).bool(), 0.0)  # 8*512
        # sum weighted sources (8*1*512 X 8*512*1024=8*1*1024)，然后压缩一个维度变成了8*1024
        # 再用encoder的隐藏状态乘以sotmax之后的得分
        batch_outputs = torch.bmm(masked_attn_scores.unsqueeze(1), key).squeeze(1)  # b x hidden

        return batch_outputs, attn_scores


def my_linear(i_dim, o_dim, bias=True):
    m = nn.Linear(i_dim, o_dim, bias)
    nn.init.normal_(m.weight, std=0.02)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


class Bert_model(nn.Module):
    def __init__(self, args):
        super(Bert_model, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_path)
        for param in self.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(args.bert_embedding, args.rnn_hidden, args.num_layers,
                            bidirectional=True, batch_first=True, dropout=args.dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.attention = Attention(args.rnn_hidden * 2)
        self.mylayer = nn.ModuleList()
        for _ in range(self.bert.config.num_hidden_layers):
            self.mylayer.append(copy.deepcopy(my_linear(args.bert_embedding, 1)))
        self.linear1 = nn.Linear(args.bert_embedding, args.rnn_hidden)
        self.fc_rnn = nn.Linear(args.rnn_hidden * 2, args.num_classes)
        self.linear2 = nn.Sequential(
            nn.Linear(args.rnn_hidden * 2, args.rnn_hidden),  # 这里是rnn_hidden * 2是因为使用了BiLSTM
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.rnn_hidden, args.num_classes)
        )

    def forward(self, inputs):
        contents = inputs[0]
        segments = inputs[1]
        masks = inputs[2]

        contents = torch.squeeze(contents, dim=1)
        segments = torch.squeeze(segments, dim=1)
        masks = torch.squeeze(masks, dim=1)

        encoder_out, text_cls = self.bert(input_ids=contents, token_type_ids=segments,
                                          attention_mask=masks, output_all_encoded_layers=True)
        # ---------------------------bert dynamic-------------------------------
        layer_logits = []
        # encoder_out 一个list,里面有12个tensor，每一个都是 [batch_size, max_len, embedding_dim]
        for i, layer_module in enumerate(self.mylayer):
            layer_logits.append(layer_module(encoder_out[i]))
        # print("np.array(layer_logits).shape:", np.array(layer_logits).shape)
        layer_logits = torch.cat(layer_logits, dim=2)  # 第三维度拼接[batchsize, max_len, 12]
        # print("layer_logits.shape:", layer_logits.shape)
        layer_dist = nn.functional.softmax(layer_logits)  # [batchszie, max_len, 12]
        # print("layer_dist.shape:", layer_dist.shape)
        # [batchsize,max_len,12,768] 这里应该可以用torch.stack((encoder_out), axis=2) 直接得到
        seq_out = torch.cat([torch.unsqueeze(x, dim=2) for x in encoder_out], dim=2)
        # print("seq_out.shape:", seq_out.shape)
        # [batchsize,max_len,1,12] × [batchsize,max_len,12,768]([1,12] × [12,768] = [1, 768])
        pooled_output = torch.matmul(torch.unsqueeze(layer_dist, dim=2), seq_out)  # pooled_output = [batchsize, max_len, 1, 768]
        pooled_output = torch.squeeze(pooled_output, dim=2)  # 再压缩回来 [batchsize, max_len, 768]
        # print("pooled_output.shape:", pooled_output.shape)
        pooled_layer = pooled_output
        # print("=============================")
        # output_layer = self.linear1(pooled_layer)
        # out, _ = self.lstm(output_layer)
        # ---------------------------bert dynamic-------------------------------
        # output_layer = self.dropout(pooled_layer)
        # lstm_hidden, _ = self.lstm(pooled_layer)
        # lstm_hiddens = lstm_hidden * masks.unsqueeze(2)
        # out, atten_scores = self.attention(lstm_hiddens, masks)
        # out = self.dropout(out)
        # logits = self.linear2(out)
        # probs = nn.functional.softmax(logits, dim=1)

        out, _ = self.lstm(pooled_layer)
        out = self.dropout(out)
        logits = self.fc_rnn(out[:, -1, :])
        probs = nn.functional.softmax(logits, dim=1)
        return logits, probs
