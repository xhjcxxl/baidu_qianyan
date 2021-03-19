import torch
import torch.nn as nn
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

def generate_sent_masks(enc_hiddens, source_lengths):
    """
    这里是用在
    :param enc_hiddens:
    :param source_lengths:
    :return:
    """

    enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
    for e_id, src_len in enumerate(source_lengths):
        enc_masks[e_id, :src_len] = 1
    return enc_masks


def corrcet_predictions(probs, targets):
    _, out_classes = probs.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


def train(model, dataloader, optimizer, criterion, args):
    model.train()
    device = args.device
    epoch_start = time.time()

    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0

    tqdm_batch_iterator = tqdm(dataloader)  # 这个是为了后面进行描述用的
    for batch_index, (batch_token_ids, batch_segment_ids, batch_mask_ids, batch_label_ids) in enumerate(tqdm_batch_iterator):
        batch_start = time.time()

        batch_token_ids = batch_token_ids.to(device)
        batch_segment_ids = batch_segment_ids.to(device)
        batch_mask_ids = batch_mask_ids.to(device)
        labels = batch_label_ids.to(device)

        labels = torch.squeeze(labels, dim=1)

        outputs, probs = model([batch_token_ids, batch_segment_ids, batch_mask_ids])
        loss = criterion(outputs, labels)
        loss.backward()
        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # 统计相关相关信息，并打印描述信息
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += corrcet_predictions(probs, labels)
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_avg / (batch_index + 1), running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, dataloader, criterion, args):
    model.eval()
    device = args.device
    epoch_start = time.time()

    batch_time_avg = 0.0
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []

    with torch.no_grad():
        for (batch_token_ids, batch_segment_ids, batch_mask_ids, batch_label_ids) in dataloader:
            batch_token_ids = batch_token_ids.to(device)
            batch_segment_ids = batch_segment_ids.to(device)
            batch_mask_ids = batch_mask_ids.to(device)
            labels = batch_label_ids.to(device)

            labels = torch.squeeze(labels, dim=1)

            outputs, probs = model([batch_token_ids, batch_segment_ids, batch_mask_ids])
            loss = criterion(outputs, labels)

            # 统计相关相关信息，并打印描述信息
            running_loss += loss.item()
            running_accuracy += corrcet_predictions(probs, labels)
            all_prob.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(batch_label_ids)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy, roc_auc_score(all_labels, all_prob)


def predict(model, dataloader, args):
    """
    用于无标签的预测，用于提交结果的
    :param model:
    :param dataloader:
    :return:
    """
    model.load_state_dict(torch.load(args.save_path)["model"])
    # Switch the model to eval mode.
    model.eval()
    device = args.device
    predict_all = np.array([], dtype=int)
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for (batch_token_ids, batch_segment_ids, batch_mask_ids, batch_label_ids) in dataloader:
            # Move input and output data to the GPU if one is used.
            batch_token_ids = batch_token_ids.to(device)
            batch_segment_ids = batch_segment_ids.to(device)
            batch_mask_ids = batch_mask_ids.to(device)
            # batch_label_ids = batch_label_ids.to(device)
            _, probs = model([batch_token_ids, batch_segment_ids, batch_mask_ids])
            predict = torch.max(probs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predict)
    return predict_all
