import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import argparse
import torch
from torch.utils.data import DataLoader
from data import LCQMC_dataset, load_embeddings
from utils import predict
from model import ESIM
import torch.nn as nn
import numpy as np
import pandas as pd


def main(args):
    print(20 * "=", " Preparing for training ", 20 * "=")
    if not os.path.exists(args.result):
        os.makedirs(args.result)

    # -------------------- Loda pretraining model ------------------- #
    checkpoints = torch.load(args.pretrained_file)
    # 可以从模型中直接恢复，也可以直接在前面定义 Retrieving model parameters from checkpoint.
    # hidden_size = checkpoints["model"]["projection.0.weight"].size(0)
    # num_classes = checkpoints["model"]["classification.6.weight"].size(0)
    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    test_data = LCQMC_dataset(args.test_file, args.vocab_file, args.max_length, test_flag=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    embeddings = load_embeddings(args.embed_file)
    model = ESIM(args, embeddings=embeddings).to(args.device)
    model.load_state_dict(checkpoints["model"])
    print(20 * "=", " Testing ESIM model on device: {} ".format(args.device), 20 * "=")
    all_predict = predict(model, test_loader)
    index = np.array([], dtype=int)
    for i in range(len(all_predict)):
        index = np.append(index, i)
    # ---------------------生成文件--------------------------
    df_test = pd.DataFrame(columns=['index', 'prediction'])
    df_test['index'] = index
    df_test['prediction'] = all_predict
    df_test.to_csv(args.submit_example_path, index=False, columns=['index', 'prediction'], sep='\t')


if __name__ == "__main__":
    all_dataset = {0: "lcqmc", 1: "bq_corpus", 2: "paws-x-zh"}
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    # parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
    args = parser.parse_args()

    using_dataset = all_dataset.get(2)
    args.embed_file = '/home/xiaxiaolin/kg/word_embedding/chinese/wikipedia_zh/token_vec_300.bin'
    args.vocab_file = '/home/xiaxiaolin/kg/word_embedding/chinese/wikipedia_zh/vocab.txt'

    args.test_file = '../dataset/' + using_dataset + '/test.tsv'
    args.result = 'result/'
    args.submit_example_path = args.result + using_dataset + '.tsv'  # 提交格式
    args.pretrained_file = 'output/' + using_dataset + '/new_best.pth.tar'

    # -------------------- 模型相关 ------------------- #
    args.max_length = 50
    args.hidden_size = 300
    args.dropout = 0.2
    args.num_classes = 2
    # -------------------- 训练相关 ------------------- #
    args.epochs = 50
    args.batch_size = 128
    args.lr = 0.0005
    args.patience = 5
    args.max_grad_norm = 10.0

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

    print("train model starting...")
    # -------------------- 预测相关 ------------------- #
    main(args)
    print("train model end...")
