import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import train, validate
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertAdam
import argparse
import numpy as np
import pandas as pd

from data import DataProcessForSentence
from utils import predict
from model import Bert_model


def main(args):
    print(20 * "=", " Preparing for training ", 20 * "=")
    if not os.path.exists(args.result):
        os.makedirs(args.result)

    tokenizer = BertTokenizer.from_pretrained(args.vocab)
    # -------------------- Data loading ------------------- #
    print("\t* Loading testing data...")
    # train_data = LCQMC_dataset(args.train_file, args.vocab_file, args.max_length, test_flag=False)
    test_data = DataProcessForSentence(tokenizer, args.test_file, args.max_length, test_flag=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    model = Bert_model(args).to(args.device)
    all_predict = predict(model, test_loader, args)
    index = np.array([], dtype=int)
    for i in range(len(all_predict)):
        index = np.append(index, i)
    # ---------------------生成文件--------------------------
    df_test = pd.DataFrame(columns=['index', 'prediction'])
    df_test['index'] = index
    df_test['prediction'] = all_predict
    df_test.to_csv(args.submit_example_path, index=False, columns=['index', 'prediction'], sep='\t')


if __name__ == "__main__":
    all_dataset = {0: "lcqmc", 1: "bq_corpus", 2: "paws-x"}
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    # parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
    args = parser.parse_args()

    using_dataset = all_dataset.get(0)
    # 数据集相关
    args.result = 'result/'
    args.test_file = '../dataset/' + using_dataset + '/test.tsv'
    args.submit_example_path = args.result + using_dataset + '.tsv'  # 提交格式
    # -------------------- 模型相关 ------------------- #
    args.bert_path = '/home/xiaxiaolin/kg/pretrained_model/bert_torch/bert_base_chiense_torch/'
    args.vocab = args.bert_path + 'vocab.txt'
    args.save_path = 'output/' + using_dataset + '/bert_best.bin'

    args.max_length = 103
    args.dropout = 0.2
    args.num_classes = 2
    args.bert_embedding = 768
    args.rnn_hidden = 512
    args.num_layers = 2

    # -------------------- 训练相关 ------------------- #
    args.epochs = 10
    args.batch_size = 32
    args.lr = 2e-05
    args.patience = 3
    args.max_grad_norm = 10.0
    args.checkpoint = None
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

    print("train model starting...")
    main(args)
    print("train model end...")
