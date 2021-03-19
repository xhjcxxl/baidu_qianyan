import re
import gensim
import pandas
import torch
from hanziconv import HanziConv
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class LCQMC_dataset(Dataset):
    def __init__(self, LCQMC_file, vocab_file, max_char_len, test_flag=False):
        word2idx, _, _ = load_vocab(vocab_file)  # 加载 词表
        # p1, p2, self.label = load_sentences(LCQMC_file, test_flag=test_flag)  # 加载源文件
        self.p1_list, self.p1_lengths, self.p2_list, self.p2_lengths, self.label = new_load_sentences(LCQMC_file, word2idx, max_char_len, test_flag=test_flag)

        # self.p1_list, self.p1_lengths, self.p2_list, self.p2_lengths = word_index(p1, p2, word2idx, max_char_len)
        self.p1_list = torch.from_numpy(self.p1_list).type(torch.long)
        self.p2_list = torch.from_numpy(self.p2_list).type(torch.long)
        self.max_length = max_char_len

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.p1_list[index], self.p1_lengths[index], self.p2_list[index], self.p2_lengths[index], self.label[index]


def get_word_list(query):
    # 对数据进行简单的过滤处理
    query = HanziConv.toSimplified(query.strip())
    regEx = re.compile('[\\W]+')  # 我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
    res = re.compile(r'([\u4e00-\u9fa5])')  # 中文范围
    sentences = regEx.split(query.lower())
    str_list = []  # 将句子划分为 单词级别
    for sentence in sentences:
        if res.split(sentence) == None:
            str_list.append(sentence)
        else:
            ret = res.split(sentence)
            str_list.extend(ret)
    return [w for w in str_list if len(w.strip()) > 0]


# 加载 word2vec embedding 矩阵
def load_embeddings(embedding_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, binary=False)
    embedding_matrix = np.zeros((len(model.index2word) + 1, model.vector_size))
    # 填充向量矩阵
    for idx, word in enumerate(model.index2word):
        embedding_matrix[idx + 1] = model[word]  # 词向量矩阵
    return embedding_matrix


def load_train_dev_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    sents1, sents2, labels = [], [], []
    with open(filename, encoding='utf-8') as f:
        for line in f:  # 遍历文件
            if '\t\t' in line:
                continue
            text1, text2, label = line.strip().split('\t')  # 获取text和label
            p1 = get_word_list(text1)
            p2 = get_word_list(text2)
            sents1.append(p1)
            sents2.append(p2)
            labels.append(int(label))
    return sents1, sents2, labels


def new_load_train_dev_data(filename, word2idx, max_char_len):
    """
    加载数据
    """
    p1_list, p1_length, p2_list, p2_length, labels = [], [], [], [], []
    with open(filename, encoding='utf-8') as f:
        for line in f:  # 遍历文件
            if '\t\t' in line:
                continue
            text1, text2, label = line.strip().split('\t')  # 获取text和label
            p1_sent = get_word_list(text1)
            p2_sent = get_word_list(text2)
            p1 = [word2idx[word] for word in p1_sent if word in word2idx.keys()]
            p2 = [word2idx[word] for word in p2_sent if word in word2idx.keys()]
            if len(p1) == 0:
                continue
            if len(p2) == 0:
                continue
            p1_list.append(p1)
            p1_length.append(min(len(p1), max_char_len))  # 超过就直接进行截断
            p2_list.append(p2)
            p2_length.append(min(len(p2), max_char_len))  # 超过截断
            labels.append(int(label))
    # padding 句子
    p1_list = pad_sequences(p1_list, maxlen=max_char_len)
    p2_list = pad_sequences(p2_list, maxlen=max_char_len)
    return p1_list, p1_length, p2_list, p2_length, labels


def new_load_test_data(filename, word2idx, max_char_len):
    """
    加载数据
    """
    p1_list, p1_length, p2_list, p2_length, labels = [], [], [], [], []
    with open(filename, encoding='utf-8') as f:
        for line in f:  # 遍历文件
            if '\t\t' in line:
                continue
            text1, text2 = line.strip().split('\t')  # 获取text和label
            p1_sent = get_word_list(text1)
            p2_sent = get_word_list(text2)
            p1 = [word2idx[word] for word in p1_sent if word in word2idx.keys()]
            p2 = [word2idx[word] for word in p2_sent if word in word2idx.keys()]
            if len(p1) == 0:
                continue
            if len(p2) == 0:
                continue
            p1_list.append(p1)
            p1_length.append(min(len(p1), max_char_len))  # 超过就直接进行截断
            p2_list.append(p2)
            p2_length.append(min(len(p2), max_char_len))  # 超过截断
            labels.append(int(0))
    # padding 句子
    p1_list = pad_sequences(p1_list, maxlen=max_char_len)
    p2_list = pad_sequences(p2_list, maxlen=max_char_len)
    return p1_list, p1_length, p2_list, p2_length, labels


def load_test_data(filename):
    sents1, sents2, labels = [], [], []
    with open(filename, encoding='utf-8') as f:
        for line in f:  # 遍历文件
            if '\t\t' in line:
                continue
            text1, text2 = line.strip().split('\t')  # 获取text和label
            p1 = get_word_list(text1)
            p2 = get_word_list(text2)
            sents1.append(p1)
            sents2.append(p2)
            labels.append(int(0))
    return sents1, sents2, labels


def load_sentences(file, test_flag=False):

    if test_flag:
        return load_test_data(file)
    else:
        return load_train_dev_data(file)


def new_load_sentences(file, word2idx, max_char_len, test_flag=False):
    if test_flag:
        return new_load_test_data(file, word2idx, max_char_len)
    else:
        return new_load_train_dev_data(file, word2idx, max_char_len)


def load_vocab(vocab_file):
    vocab = [line.strip() for line in open(vocab_file, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}

    return word2idx, idx2word, vocab


def word_index(p1_sentences, p2_sentences, word2idx, max_char_len):
    p1_list, p1_length, p2_list, p2_length = [], [], [], []
    for p1_sent, p2_sent in zip(p1_sentences, p2_sentences):
        # 句子 转为 索引
        p1 = [word2idx[word] for word in p1_sent if word in word2idx.keys()]
        p2 = [word2idx[word] for word in p2_sent if word in word2idx.keys()]
        p1_list.append(p1)
        p1_length.append(min(len(p1), max_char_len))
        p2_list.append(p2)
        p2_length.append(min(len(p2), max_char_len))

    # padding 句子
    p1_list = pad_sequences(p1_list, maxlen=max_char_len)
    p2_list = pad_sequences(p2_list, maxlen=max_char_len)
    return p1_list, p1_length, p2_list, p2_length


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences
    把序列长度转变为一样长的，如果设置了maxlen则长度统一为maxlen，如果没有设置则默认取
    最大的长度。填充和截取包括两种方法，post与pre，post指从尾部开始处理，pre指从头部
    开始处理，默认都是从尾部开始。
    Arguments:
        sequences: 序列
        maxlen: int 最大长度
        dtype: 转变后的数据类型
        padding: 填充方法'pre' or 'post'
        truncating: 截取方法'pre' or 'post'
        value: float 填充的值
    Returns:
        x: numpy array 填充后的序列维度为 (number_of_sequences, maxlen)
    """
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x
