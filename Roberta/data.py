from torch.utils.data import Dataset
from hanziconv import HanziConv
import pandas as pd
import torch


class DataProcessForSentence(Dataset):
    def __init__(self, bert_tokenizer, input_file, max_len=103, test_flag=False):
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_len
        self.p1_list, self.p2_list, self.labels = self.get_input(input_file, test_flag)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # padding
        token_ids, segment_ids, mask_ids = self.trunate_and_pad(self.p1_list[index], self.p2_list[index])
        label_ids = self.labels[index]

        token_ids = torch.LongTensor([token_ids]).type(torch.long)
        segment_ids = torch.LongTensor([segment_ids]).type(torch.long)
        mask_ids = torch.LongTensor([mask_ids]).type(torch.long)
        label_ids = torch.LongTensor([label_ids]).type(torch.long)
        return token_ids, segment_ids, mask_ids, label_ids

    def get_input(self, file, test_flag=False):
        """
        对输入文本进行简单的处理，进行简单的分词进行判断，因为存在某些词分出来没有对应的token的
        :param file:
        :param test_flag:
        :return:
        """
        p1_list, p2_list, labels = [], [], []
        if test_flag:
            with open(file, encoding='utf-8') as f:
                for line in f:
                    if '\t\t' in line:
                        continue
                    text1, text2 = line.strip().split('\t')
                    # 化为简体字
                    text1 = HanziConv.toSimplified(text1)
                    text2 = HanziConv.toSimplified(text2)
                    # 分词
                    tokens_seq_1 = list(self.bert_tokenizer.tokenize(text1))
                    tokens_seq_2 = list(self.bert_tokenizer.tokenize(text2))
                    if len(tokens_seq_1) == 0:
                        continue
                    if len(tokens_seq_2) == 0:
                        continue
                    p1_list.append(tokens_seq_1)
                    p2_list.append(tokens_seq_2)
                    labels.append(int(0))
            return p1_list, p2_list, labels

        else:
            with open(file, encoding='utf-8') as f:
                for line in f:
                    if '\t\t' in line:
                        continue
                    text1, text2, label = line.strip().split('\t')
                    # 化为简体字
                    text1 = HanziConv.toSimplified(text1)
                    text2 = HanziConv.toSimplified(text2)
                    # 分词
                    tokens_seq_1 = list(self.bert_tokenizer.tokenize(text1))
                    tokens_seq_2 = list(self.bert_tokenizer.tokenize(text2))
                    if len(tokens_seq_1) == 0:
                        continue
                    if len(tokens_seq_2) == 0:
                        continue
                    p1_list.append(tokens_seq_1)
                    p2_list.append(tokens_seq_2)
                    labels.append(int(label))
            return p1_list, p2_list, labels

    def trunate_and_pad(self, tokens_seq_1, tokens_seq_2):
        """
        1. 如果是单句序列，按照BERT中的序列处理方式，需要在输入序列头尾分别拼接特殊字符'CLS'与'SEP'，
           因此不包含两个特殊字符的序列长度应该小于等于max_seq_len-2，如果序列长度大于该值需要那么进行截断。
        2. 对输入的序列 最终形成['CLS',seq,'SEP']的序列，该序列的长度如果小于max_seq_len，那么使用0进行填充。
        入参:
            seq_1       : 输入序列，在本处其为单个句子。
            seq_2       : 输入序列，在本处其为单个句子。
            max_seq_len : 拼接'CLS'与'SEP'这两个特殊字符后的序列长度

        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，单句，取值都为0 ，双句按照01切分
        """
        # 对超长序列进行截断 这里-3是因为两个句子使用了 CLS SEP SEP，除以二是只取一半，句子没那么长
        if len(tokens_seq_1) > ((self.max_seq_len - 3) // 2):
            tokens_seq_1 = tokens_seq_1[0:(self.max_seq_len - 3) // 2]
        if len(tokens_seq_2) > ((self.max_seq_len - 3) // 2):
            tokens_seq_2 = tokens_seq_2[0:(self.max_seq_len - 3) // 2]
        # 分别在首尾拼接特殊符号
        tokens = ['[CLS]'] + tokens_seq_1 + ['[SEP]'] + tokens_seq_2 + ['[SEP]']
        segments = [0] * (len(tokens_seq_1) + 2) + [1] * (len(tokens_seq_2) + 1)

        # to ids
        token_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        # padding
        padding = [0] * (self.max_seq_len - len(token_ids))

        mask_ids = [1] * len(token_ids) + padding
        segment_ids = segments + padding
        token_ids = token_ids + padding

        assert len(token_ids) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len
        assert len(mask_ids) == self.max_seq_len
        return token_ids, segment_ids, mask_ids
