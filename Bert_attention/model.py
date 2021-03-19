import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
import torch.nn.functional as F


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
        # self.fc_rnn = nn.Linear(args.rnn_hidden * 2, args.num_classes)
        self.linear = nn.Sequential(
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
                                          attention_mask=masks, output_all_encoded_layers=False)
        lstm_hidden, _ = self.lstm(encoder_out)
        lstm_hiddens = lstm_hidden * masks.unsqueeze(2)
        out, atten_scores = self.attention(lstm_hiddens, masks)
        out = self.dropout(out)
        logits = self.linear(out)
        probs = nn.functional.softmax(logits, dim=1)
        return logits, probs
