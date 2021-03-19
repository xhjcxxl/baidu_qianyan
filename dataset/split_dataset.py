import pandas as pd
import numpy as np
import codecs
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


def skf_split_csv(infile, train_file, valid_file, seed=999, ratio=0.2):
    infile_df = pd.read_csv(infile, sep='\t', encoding='utf-8', header=None)
    print(infile_df.head())
    print(infile_df.shape)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1314)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(infile_df[0], infile_df[2]), 1):
        print(f'Fold {fold}')
        infile_df.iloc[train_idx, :].to_csv(train_file + f'_{fold}.tsv', sep='\t', encoding='utf-8', index=False, header=None)
        infile_df.iloc[valid_idx, :].to_csv(valid_file + f'_{fold}.tsv', sep='\t', encoding='utf-8', index=False, header=None)


all_dataset = {0: "lcqmc", 1: "bq_corpus", 2: "paws-x"}
using_dataset = all_dataset.get(2)
merge_file = using_dataset + '/merge_train.tsv'
train_file = using_dataset + '/split_train'
dev_file = using_dataset + '/split_dev'

skf_split_csv(infile=merge_file, train_file=train_file, valid_file=dev_file)
print("ok")
