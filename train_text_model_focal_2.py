# coding: UTF-8
import time
import torch
import numpy as np
from importlib import import_module
import argparse
from util.utils import get_time_dif
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import scipy

from model.text_model_focal_2 import TEXTNET
from util.dataset_text import BaseDataset
import torch.nn.functional as F
from model.pytorch_pretrained.optimization import BertAdam

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True  

lr = 5e-5
max_epoch = 20
batch_size = 256


def write_log(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    filename = log_dir + '/text_model_com_focal_2_without_pair.log'
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
    )
    return logging


def net_train(fold, net, train_loader, dev_iter):
    start_time = time.time()
    param_optimizer = list(net.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]


    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=lr,
                         warmup=0.05,
                         t_total=len(train_loader) * max_epoch)
    total_batch = 0 
    dev_best_acc = 0.
    num_true = 0
    num_pred = 0
    num_target = 0
    net.train()
    for epoch in range(max_epoch):
        print('Epoch [{}/{}]'.format(epoch + 1, max_epoch))
        logging.info('Epoch [{}/{}]'.format(epoch + 1, max_epoch))
        net.train()
        for i, (vid, text, tag) in enumerate(train_loader):
            tag = tag.to(device)
            input_id = text[0].to(device)
            token_type_ids = text[1].to(device)
            attention_mask = text[2].to(device)
            loss, prob, _ = net((input_id, token_type_ids, attention_mask), tag)
            num_true, num_pred, num_target = cal_metrics(num_true, num_pred, num_target, tag, prob)
            loss.backward()
            optimizer.step()
            net.zero_grad()
            if total_batch % 50 == 0:
                precision = num_true / (num_pred + 1e-6)
                recall = num_true / (num_target + 1e-6)
                num_true = 0
                num_pred = 0
                num_target = 0
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.4}, Train Precision: {2:>6.4%}, Train Recall: {3:>6.4%}, Time: {4}'
                print(msg.format(total_batch, loss.item(), precision, recall, time_dif))
                logging.info(msg.format(total_batch, loss.item(), precision, recall, time_dif))
            total_batch += 1
        print('----------------------------Evaluate----------------------------')
        logging.info('----------------------------Evaluate----------------------------')
        dev_loss, dev_acc = evaluate_spearman(net, dev_iter)
        time_dif = get_time_dif(start_time)
        msg = 'Val Loss: {0:>5.4}, Val ACC: {1:>6.4%},  Time: {2}'
        print(msg.format(dev_loss, dev_acc, time_dif))
        logging.info(msg.format(dev_loss, dev_acc, time_dif))
        if dev_best_acc < dev_acc:
            print('!!!!!! Best model save!!!!!!!!!!!!')
            logging.info('!!!!!! Best model save!!!!!!!!!!!!')
            torch.save(net.state_dict(), 'checkpoint/pretrain_model/text_model_com_focal_2_without_pair.pth') 
            dev_best_acc = dev_acc
    return net

def cal_metrics(num_true, num_pred, num_target, labels, logits):
    labels = labels.detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    pred = logits > 0.5
    num_true += ((labels == pred) * labels).sum()
    num_pred += pred.sum()
    num_target += labels.sum()
    return num_true, num_pred, num_target

        

def evaluate_spearman(net, data_loader):
    net.eval()
    id_list = []
    loss_total = 0
    embedding_list = []
    with torch.no_grad():
        for vid, text,tag in data_loader:
            ids = vid
            tag = tag.to(device)
            input_id = text[0].to(device)
            token_type_ids = text[1].to(device)
            attention_mask = text[2].to(device)
            loss, prob, embedding = net((input_id, token_type_ids, attention_mask), tag)
            loss_val = loss.item()
            loss_total += loss_val
            embedding = F.normalize(embedding, p=2, dim=1)
            embedding = embedding.detach().cpu().numpy()
            embedding_list.append(embedding)
            id_list += ids
    embeddings = np.concatenate(embedding_list)
    embedding_map = dict(zip(id_list, embeddings))
    annotate = {}
    label_file = 'data/pairwise/label.tsv'
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            rk1, rk2, score = line.split('\t')
            annotate[(rk1, rk2)] = float(score)
    sim_res = []
    for (k1, k2), v in annotate.items():
        if k1 not in embedding_map or k2 not in embedding_map:
            continue
        sim_res.append((v, (embedding_map[k1] * embedding_map[k2]).sum()))
    spearman = scipy.stats.spearmanr([x[0] for x in sim_res], [x[1] for x in sim_res]).correlation
    return loss_total/ len(data_loader), spearman


if __name__ == "__main__":
    logging = write_log('logs')
    train_data_path_list=['data/pointwise/pretrain_0.tfrecords'
    ,'data/pointwise/pretrain_1.tfrecords'
    ,'data/pointwise/pretrain_2.tfrecords'
    ,'data/pointwise/pretrain_3.tfrecords'
    ,'data/pointwise/pretrain_4.tfrecords'
    ,'data/pointwise/pretrain_5.tfrecords'
    ,'data/pointwise/pretrain_6.tfrecords'
    ,'data/pointwise/pretrain_7.tfrecords'
    ,'data/pointwise/pretrain_8.tfrecords'
    ,'data/pointwise/pretrain_9.tfrecords'
    ,'data/pointwise/pretrain_10.tfrecords'
    ,'data/pointwise/pretrain_11.tfrecords'
    ,'data/pointwise/pretrain_12.tfrecords'
    ,'data/pointwise/pretrain_13.tfrecords'
    ,'data/pointwise/pretrain_14.tfrecords'
    ,'data/pointwise/pretrain_15.tfrecords'
    ,'data/pointwise/pretrain_16.tfrecords'
    ,'data/pointwise/pretrain_17.tfrecords'
    ,'data/pointwise/pretrain_18.tfrecords'
    ,'data/pointwise/pretrain_19.tfrecords']

    eval_data_path_list=['data/pairwise/pairwise.tfrecords']
    desc_file = 'data/desc.json'

    train_dataset = BaseDataset(train_data_path_list, desc_file, text_type='com_text', training=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=8,pin_memory=True, shuffle=True)

    eval_dataset = BaseDataset(eval_data_path_list, desc_file, text_type='com_text', training=True)

    eval_loader = DataLoader(eval_dataset, batch_size=batch_size,num_workers=8,pin_memory=True, shuffle=False)

    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)

    net = TEXTNET().to(device)
    net = net_train(0,net=net, train_loader=train_loader, dev_iter=eval_loader)


