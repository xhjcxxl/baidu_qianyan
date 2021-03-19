import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import train, validate
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertAdam
import argparse

from data import DataProcessForSentence
from utils import train, validate
from model import Bert_model


def main(args):
    print(20 * "=", " Preparing for training ", 20 * "=")
    # 保存模型的路径
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    tokenizer = BertTokenizer.from_pretrained(args.vocab)
    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    # train_data = LCQMC_dataset(args.train_file, args.vocab_file, args.max_length, test_flag=False)
    train_data = DataProcessForSentence(tokenizer, args.train_file, args.max_length, test_flag=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    print("\t* Loading valid data...")
    dev_data = DataProcessForSentence(tokenizer, args.dev_file, args.max_length, test_flag=False)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=True)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    model = Bert_model(args).to(args.device)

    # -------------------- Preparation for training  ------------------- #
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    # 列出所有需要更新权重的参数
    param_optimizer = list(model.named_parameters())
    # 不需要权重衰减的
    no_decay = ['bias', 'LearyNorm.bias', 'LayerNorm.weight']
    # 不是这几种类型的就需要进行权重衰减，这里中是不需要进行权重衰减的，保持正常的梯度更新即可
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.00}]

    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=args.lr,
        warmup=0.5,
        t_total=len(train_loader) * args.epochs)

    # 学习计划
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.85, patience=0)

    best_score = 0.0
    start_epoch = 1

    epochs_count = []
    train_losses = []
    valid_losses = []
    # Continuing training from a checkpoint if one was given as argument
    if args.checkpoint:
        # 从文件中加载checkpoint数据, 从而继续训练模型
        checkpoints = torch.load(args.checkpoint)
        start_epoch = checkpoints["epoch"] + 1
        best_score = checkpoints["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoints["model"])  # 模型部分
        optimizer.load_state_dict(checkpoints["optimizer"])
        epochs_count = checkpoints["epochs_count"]
        train_losses = checkpoints["train_losses"]
        valid_losses = checkpoints["valid_losses"]

        # 这里改为只有从以前加载的checkpoint中才进行计算 valid, Compute loss and accuracy before starting (or resuming) training.
        _, valid_loss, valid_accuracy, auc = validate(model, dev_loader, criterion)
        print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}"
              .format(valid_loss, (valid_accuracy * 100), auc))
    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training Bert model on device: {}".format(args.device), 20 * "=")
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):
        epochs_count.append(epoch)
        # -------------------- train --------------------------
        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer, criterion, args)
        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))

        # -------------------- valid --------------------------
        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy, epoch_auc = validate(model, train_loader, criterion, args)
        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100), epoch_auc))

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)
        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            # 保存最好的结果，需要保存的参数，这些参数在checkpoint中都能找到
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "epochs_count": epochs_count,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses},
                os.path.join(args.target_dir, "bert_best.bin"))
        if patience_counter >= args.patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break


if __name__ == "__main__":
    all_dataset = {0: "lcqmc", 1: "bq_corpus", 2: "paws-x"}
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    # parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
    args = parser.parse_args()

    using_dataset = all_dataset.get(0)
    # 数据集相关
    # args.train_file = '../dataset/' + using_dataset + '/new_train.tsv'
    args.train_file = '../dataset/' + using_dataset + '/new_train.tsv'
    args.dev_file = '../dataset/' + using_dataset + '/dev.tsv'
    args.target_dir = 'output/' + using_dataset
    # -------------------- 模型相关 ------------------- #
    args.bert_path = '/home/xiaxiaolin/kg/pretrained_model/bert_torch/bert_base_chiense_torch/'
    args.vocab = args.bert_path + 'vocab.txt'

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

    print("{} train model starting...".format(using_dataset))
    main(args)
    print("train model end...")
