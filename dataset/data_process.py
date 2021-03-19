import numpy as np
import pandas as pd

all_dataset = {0: "lcqmc", 1: "bq_corpus", 2: "paws-x-zh"}
using_dataset = all_dataset.get(2)

train_file = using_dataset + '/train.tsv'
dev_file = using_dataset + '/dev.tsv'

# df_train = pd.read_csv(train_file, sep='\t', encoding='utf-8', header=None, error_bad_lines=False)

# sentences = '		0'
# data = sentences.split('\t')
# print(data)
#
D = []
with open(train_file, encoding='utf-8') as f:
    for line in f:  # 遍历文件
        if '\t\t' in line:
            continue
        text1, text2, label = line.strip().split('\t')  # 获取text和label
        if len(text1) < 0:
            print(text1)
        if len(text2) < 0:
            print(text2)
        D.append((text1, text2, int(label)))  # 保存为list形式

print("ok")
