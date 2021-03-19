import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import argparse
import torch
from torch.utils.data import DataLoader
from data import LCQMC_dataset, load_embeddings
from utils import train, validate
from model import ESIM
import torch.nn as nn


def main(args):
    print(20 * "=", " Preparing for training ", 20 * "=")
    # 保存模型的路径
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    # train_data = LCQMC_dataset(args.train_file, args.vocab_file, args.max_length, test_flag=False)
    train_data = LCQMC_dataset(args.train_file, args.vocab_file, args.max_length, test_flag=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    print("\t* Loading valid data...")
    dev_data = LCQMC_dataset(args.dev_file, args.vocab_file, args.max_length, test_flag=False)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=True)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    embeddings = load_embeddings(args.embed_file)
    model = ESIM(args, embeddings=embeddings).to(args.device)

    # -------------------- Preparation for training  ------------------- #
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    # 过滤出需要梯度更新的参数
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)  # 优化器
    # 学习计划
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.85, patience=0)

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
    print("\n", 20 * "=", "Training ESIM model on device: {}".format(args.device), 20 * "=")
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):
        epochs_count.append(epoch)
        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer,
                                                       criterion, epoch, args.max_grad_norm)
        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))

        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy, epoch_auc = validate(model, train_loader, criterion)
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
                os.path.join(args.target_dir, "new_best.pth.tar"))
        # 保存每个epoch的结果 Save the model at each epoch.(这里可要可不要)
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "best_score": best_score,
                "optimizer": optimizer.state_dict(),
                "epochs_count": epochs_count,
                "train_losses": train_losses,
                "valid_losses": valid_losses},
            os.path.join(args.target_dir, "new_esim_{}.pth.tar".format(epoch)))

        if patience_counter >= args.patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break


if __name__ == "__main__":
    all_dataset = {0: "lcqmc", 1: "bq_corpus", 2: "paws-x"}
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    # parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
    args = parser.parse_args()

    using_dataset = all_dataset.get(2)

    args.train_file = '../dataset/' + using_dataset + '/new_train.tsv'
    args.dev_file = '../dataset/' + using_dataset + '/dev.tsv'
    args.embed_file = '/home/xiaxiaolin/kg/word_embedding/chinese/wikipedia_zh/token_vec_300.bin'
    args.vocab_file = '/home/xiaxiaolin/kg/word_embedding/chinese/wikipedia_zh/vocab.txt'
    args.target_dir = 'output/' + using_dataset
    # -------------------- 模型相关 ------------------- #
    args.max_length = 50
    args.hidden_size = 300
    args.dropout = 0.2
    args.num_classes = 2
    # -------------------- 训练相关 ------------------- #
    args.epochs = 50
    args.batch_size = 256
    args.lr = 0.0005
    args.patience = 5
    args.max_grad_norm = 10.0
    # args.gpu_index = 0
    args.checkpoint = None
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

    print("train model starting...")
    main(args)
    print("train model end...")
