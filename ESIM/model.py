import torch
import torch.nn as nn
from utils import get_mask, replace_masked
from layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention

"""
定义网络时，需要继承nn.Module，并实现它的forward方法。
把网络中具有可学习参数的层放在构造函数__init__中。
不具有学习参数的放在forward中使用nn.functional代替
"""

class ESIM(nn.Module):
    def __init__(self, args, embeddings=None):
        super(ESIM, self).__init__()
        self.emb = embeddings
        self.embedding_dim = self.emb.shape[1]
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.dropout = args.dropout
        self.device = args.device

        self.word_embedding = nn.Embedding(self.emb.shape[0], self.emb.shape[1])
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(self.emb))
        self.word_embedding.float()
        self.word_embedding.weight.requires_grad = True  # 可调
        self.word_embedding.to(self.device)
        if self.dropout:
            self.rnn_dropout = RNNDropout(p=self.dropout)

        # first bilstm
        self.first_rnn = Seq2SeqEncoder(nn.LSTM, self.embedding_dim, self.hidden_size, bidirectional=True)
        self.projection = nn.Sequential(
            nn.Linear(4*2*self.hidden_size, self.hidden_size),
            nn.ReLU()
        )
        self.attention = SoftmaxAttention()

        # second bilstm
        self.second_rnn = Seq2SeqEncoder(nn.LSTM, self.hidden_size, self.hidden_size, bidirectional=True)
        self.classification = nn.Sequential(
            nn.Linear(2*4*self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size//2, self.num_classes)
        )

    def forward(self, q1, q1_lengths, q2, q2_lengths):

        # 获取 mask, 用于后面 的 attention
        q1_mask = get_mask(q1, q1_lengths).to(self.device)
        q2_mask = get_mask(q2, q2_lengths).to(self.device)

        # 获取 embedding
        q1_embed = self.word_embedding(q1)
        q2_embed = self.word_embedding(q2)

        # 对 embedding 进行dropout
        if self.dropout:
            q1_embed = self.rnn_dropout(q1_embed)
            q2_embed = self.rnn_dropout(q2_embed)

        # bilstm
        try:
            q1_encoded = self.first_rnn(q1_embed, q1_lengths)
            q2_encoded = self.first_rnn(q2_embed, q2_lengths)
        except:
            print(q1_lengths, q2_lengths)
        # attention
        q1_aligned, q2_aligned = self.attention(q1_encoded, q1_mask, q2_encoded, q2_mask)

        # concat
        q1_combined = torch.cat([q1_encoded, q1_aligned, q1_encoded - q1_aligned, q1_encoded * q1_aligned], dim=-1)
        q2_combined = torch.cat([q2_encoded, q2_aligned, q2_encoded - q2_aligned, q2_encoded * q2_aligned], dim=-1)

        # 映射,降低维度
        projected_q1 = self.projection(q1_combined)
        projected_q2 = self.projection(q2_combined)

        if self.dropout:
            projected_q1 = self.rnn_dropout(projected_q1)
            projected_q2 = self.rnn_dropout(projected_q2)

        # 再一次 rnn,使用 lstm
        q1_compare = self.second_rnn(projected_q1, q1_lengths)
        q2_compare = self.second_rnn(projected_q2, q2_lengths)

        # 平均池化 + 最大池化
        q1_avg_pool = torch.sum(q1_compare * q1_mask.unsqueeze(1).transpose(2, 1), dim=1)/torch.sum(q1_mask, dim=1, keepdim=True)
        q2_avg_pool = torch.sum(q2_compare * q2_mask.unsqueeze(1).transpose(2, 1), dim=1)/torch.sum(q2_mask, dim=1, keepdim=True)

        q1_max_pool, _ = replace_masked(q1_compare, q1_mask, -1e7).max(dim=1)
        q2_max_pool, _ = replace_masked(q2_compare, q2_mask, -1e7).max(dim=1)

        # 拼接成最后的特征向量
        merged = torch.cat([q1_avg_pool, q1_max_pool, q2_avg_pool, q2_max_pool], dim=1)

        # 分类
        logits = self.classification(merged)
        probs = nn.functional.softmax(logits, dim=-1)
        return logits, probs
