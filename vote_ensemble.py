import pandas as pd
import random
import numpy as np
import os
model_name = ['Bert', 'Bert_Adversarial', 'Bert_attention', 'Bert_dynamic', 'ESIM', 'Nezha', 'Roberta']
all_dataset = {0: "lcqmc", 1: "bq_corpus", 2: "paws-x"}
using_dataset = all_dataset.get(2)
predict_files = '/result/' + using_dataset

# train_k = pd.read_csv('submission_beike_0.8163005990197858.tsv', sep='\t', header=None)  # roberta
# train_k.columns = ['index', 'prediction']
# a1 = train_k['prediction'].tolist()  # 需要进行对比的 主文本


def get_count(l):
    return l.count(1)


submit_example_path = 'ensemble_result/' + using_dataset + '.tsv'
labels = []  # 存储voting前的结果
label = []  # 存储voting后的结果
index = np.array([], dtype=int)
for one_model in model_name:  # 需要集成的labels
    to_read_files = one_model + predict_files + '.tsv'
    train = pd.read_csv(to_read_files, sep='\t')
    train.columns = ['index', 'prediction']
    labels.append(train['prediction'].tolist())
# c021 = 0  # 0转为1
# c120 = 0  # 1转为0
for j in range(2000):  # 总的 label个数
    each_predict = []
    for i in range(len(labels)):  # 需要ensemble的文件个数
        each_predict.append(labels[i][j])  # 对每个文件中的同一个label进行统计

    if each_predict.count(1) > 3:  # 如果为真的有8个，那么就把这个改成
        # print(j+1)
        # c021 += 1  # 统计转换的个数
        label.append(1)  # 最终这个 label 设置为1
        index = np.append(index, j)
    else:
        # print(j + 1)
        # c120 += 1  # 统计转换的个数
        label.append(0)  # 最终这个 label 设置为0
        index = np.append(index, j)
    count1 = get_count(each_predict)  # 统计个数
    # if count1 >= 5:
    #     label.append(1)
    # else:
    #     label.append(0)
# ---------------------生成文件--------------------------
df_test = pd.DataFrame(columns=['index', 'prediction'])
df_test['index'] = index
df_test['prediction'] = label
df_test.to_csv(submit_example_path, index=False, columns=['index', 'prediction'], sep='\t')


# test_left = pd.read_csv('./test/test.query.tsv', sep='\t', header=None, encoding='gbk')
# test_left.columns = ['id', 'q1']
# test_right = pd.read_csv('./test/test.reply.tsv', sep='\t', header=None, encoding='gbk')
# test_right.columns = ['id', 'id_sub', 'q2']
# df_test = test_left.merge(test_right, how='left')
# df_test.to_csv('submission_beike_{}.tsv'.format('look'), index=False, header=None,
#                                               sep='\t')
#
# df_test['label'] = np.asarray(label).astype(int)
# df_test[['id', 'id_sub', 'label']].to_csv('submission_beike_{}.tsv'.format('10vote1'), index=False, header=None, sep='\t')
