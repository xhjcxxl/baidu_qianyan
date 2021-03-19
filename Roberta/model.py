import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
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


class Bert_model(nn.Module):
    def __init__(self, args):
        super(Bert_model, self).__init__()
        self.config = BertConfig.from_pretrained(args.bert_config)
        self.bert = BertModel.from_pretrained(args.bert_path, from_tf=False, config=self.config)
        for param in self.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(args.bert_embedding, args.rnn_hidden, args.num_layers,
                            bidirectional=True, batch_first=True, dropout=args.dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.fc_rnn = nn.Linear(args.rnn_hidden * 2, args.num_classes)

    def forward(self, inputs):
        contents = inputs[0]
        segments = inputs[1]
        masks = inputs[2]

        contents = torch.squeeze(contents, dim=1)
        segments = torch.squeeze(segments, dim=1)
        masks = torch.squeeze(masks, dim=1)

        encoder_out, text_cls = self.bert(input_ids=contents, token_type_ids=segments, attention_mask=masks)
        out, _ = self.lstm(encoder_out)
        out = self.dropout(out)
        logits = self.fc_rnn(out[:, -1, :])
        probs = nn.functional.softmax(logits, dim=1)
        return logits, probs


class Roberta_pooling(nn.Module):

    def __init__(self, args):
        super(Roberta_pooling, self).__init__()
        self.config = BertConfig.from_pretrained(args.bert_config)
        self.bert = BertModel.from_pretrained(args.bert_path, from_tf=False, config=self.config)
        for param in self.parameters():
            param.requires_grad = True

        self.pooling_mode_max_tokens = True
        self.pooling_mode_mean_tokens = True
        self.lstm = nn.LSTM(args.bert_embedding, args.rnn_hidden, args.num_layers,
                            bidirectional=True, batch_first=True, dropout=args.dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.bert_embedding * 3, args.num_classes)

    def forward(self, inputs):
        contents = inputs[0]
        segments = inputs[1]
        masks = inputs[2]

        contents = torch.squeeze(contents, dim=1)
        segments = torch.squeeze(segments, dim=1)
        attention_mask = torch.squeeze(masks, dim=1)

        output_vectors = []
        outputs = self.bert(input_ids=contents, token_type_ids=segments, attention_mask=attention_mask)
        token_embeddings = outputs[0]
        # cls_token = outputs[1]
        cls_token = token_embeddings[:, 0, :]
        output_vectors.append(cls_token)  # 放到第一个，然后把mean和max的添加到后面，也就是说，这三个同时进行处理

        ## Pooling strategy
        if self.pooling_mode_max_tokens:
            # 相当于 以前是对长度进行mask标记，现在是对每个token进行mask标记
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # 一个token有768dim，也要相应设置为极小数 Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]  # 返回的是 每一个样本中哪个token值最大，一个batch就是32个，总的: 32x768
            output_vectors.append(max_over_time)

        if self.pooling_mode_mean_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)  # *是点乘，得到:32x103x768,按照维度1求和:32x768
            sum_mask = input_mask_expanded.sum(1)  # 32x103x768 按维度1求和,得: 32x768s
            sum_mask = torch.clamp(sum_mask, min=1e-9)  # 将输入input张量每个元素的夹紧到区间
            output_vectors.append(sum_embeddings / sum_mask)  # 均值 embedding  32 x 768 相当于一个token的表示结果，类似于CLS，但值完全不同

        output_vector = torch.cat(output_vectors, 1)  # 按维度拼接 768*3 =2304, 维度为 32x2304
        output_vector = self.dropout(output_vector)
        logits = self.fc(output_vector)  # 32x2
        probs = nn.functional.softmax(logits, dim=1)  # 在维度1上进行softmax

        # out, _ = self.lstm(encoder_out)
        # out = self.dropout(out)
        # logits = self.fc_rnn(out[:, -1, :])
        # probs = nn.functional.softmax(logits, dim=1)
        return logits, probs
