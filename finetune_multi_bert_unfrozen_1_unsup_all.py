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
from model.multi_bert_simnet_1 import SIMNET
from util.dataset_pair import BaseDataset
from model.pytorch_pretrained.optimization import BertAdam

                          
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True  

lr = 2e-05
max_epoch = 10
batch_size = 128

def write_log(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    filename = log_dir + '/finetune_multi_bert_all.log'
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
    )
    return logging
def get_unsup_mse_loss(embeddings):
    scores = torch.mm(embeddings, embeddings.transpose(0,1))
    unsup_mask = 1-torch.eye(scores.shape[0])
    with torch.no_grad():
        labels_unsup = torch.stack([i*torch.ones_like(scores) for i in [0. , 0.5, 1.]]).float().to(device)
        unsup_mask = unsup_mask.to(device)
    unsup_loss = F.mse_loss(scores.unsqueeze(0), labels_unsup, reduction='none').min(0)[0]
    unsup_loss = 0.5*(unsup_loss*unsup_mask).mean()
    return unsup_loss


def net_train(fold, net, train_loader):
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
    net.train()
    for epoch in range(max_epoch):
        print('Epoch [{}/{}]'.format(epoch + 1, max_epoch))
        logging.info('Epoch [{}/{}]'.format(epoch + 1, max_epoch))
        for i, (v_content_1, v_content_2, label)  in enumerate(train_loader):
            vid_1 = v_content_1[0]
            frame_feature = v_content_1[1][0][0].to(device)
            frame_mask = v_content_1[1][0][1].to(device)
            text_1 = v_content_1[1][1]

            input_id = text_1[0].to(device)
            token_type_ids = text_1[1].to(device)
            attention_mask = text_1[2].to(device)
            embedding_1 = net((input_id,token_type_ids,attention_mask), (frame_feature, frame_mask))
            embedding_1 = F.normalize(embedding_1, p=2, dim=1)

            vid_2 = v_content_2[0]
            frame_feature = v_content_2[1][0][0].to(device)
            frame_mask = v_content_2[1][0][1].to(device)

            text_2 = v_content_2[1][1]
            input_id = text_2[0].to(device)
            token_type_ids = text_2[1].to(device)
            attention_mask = text_2[2].to(device)


            embedding_2 = net((input_id, token_type_ids, attention_mask), (frame_feature, frame_mask))
            embedding_2 = F.normalize(embedding_2, p=2, dim=1)

            sim = embedding_1 * embedding_2
            sim = sim.sum(1)
            label = label.to(torch.float32).to(device)
            unsup_loss_1 = get_unsup_mse_loss(embedding_1)
            unsup_loss_2 = get_unsup_mse_loss(embedding_2)
            sup_loss = F.mse_loss(sim, label)
            loss = sup_loss + unsup_loss_1 + unsup_loss_2
            loss.backward()
            
            optimizer.step()
            net.zero_grad()
            if total_batch % 50 == 0:
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.4}, Time: {2}'
                print(msg.format(total_batch, loss.item(), time_dif))
                logging.info(msg.format(total_batch, loss.item(), time_dif))
            total_batch += 1
        torch.save(net.state_dict(), 'checkpoint/multi_bert_unfrozen_1_unsup_all_epoch_{}.pth'.format(epoch))
    return net
        

def evaluate_spearman(net, data_loader):
    net.eval()
    id_list = []
    embedding_list = []
    loss_total = 0
    with torch.no_grad():
        for i, (v_content_1, v_content_2, label)  in enumerate(data_loader):
            vid_1 = v_content_1[0]
            frame_feature = v_content_1[1][0][0].to(device)
            frame_mask = v_content_1[1][0][1].to(device)

            text_1 = v_content_1[1][1]
            input_id = text_1[0].to(device)
            token_type_ids = text_1[1].to(device)
            attention_mask = text_1[2].to(device)
            embedding_1 = net((input_id,token_type_ids,attention_mask), (frame_feature, frame_mask))
            embedding_1 = F.normalize(embedding_1, p=2, dim=1)
            embedding_list.append(embedding_1.detach().cpu().numpy())
            id_list += vid_1
            vid_2 = v_content_2[0]

            frame_feature = v_content_2[1][0][0].to(device)
            frame_mask = v_content_2[1][0][1].to(device)
            text_2 = v_content_2[1][1]
            input_id = text_2[0].to(device)
            token_type_ids = text_2[1].to(device)
            attention_mask = text_2[2].to(device)
            embedding_2 = net((input_id,token_type_ids,attention_mask), (frame_feature, frame_mask))
            embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            sim = embedding_1 * embedding_2
            sim = sim.sum(1)
            label = label.to(device)
            loss = F.mse_loss(sim, label)

            loss_total += loss
            embedding_list.append(embedding_2.detach().cpu().numpy())
            id_list += vid_2
    embeddings = np.concatenate(embedding_list)
    embedding_map = dict(zip(id_list, embeddings))
    annotate = {}
    label_file = 'data/val_data.csv'
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
    return loss_total / len(data_loader), spearman


if __name__ == "__main__":
    logging = write_log('logs')
    data_path_list='data/finetune/'
    desc_file = 'data/desc.json'
    df_train = pd.read_csv('sort_label.tsv', header=None)
    train_dataset = BaseDataset(df_train, data_path_list, text_type='title')
    train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=8,pin_memory=True, shuffle=True)
    net = SIMNET(device, ckpt='checkpoint/pretrain_model/multi_bert_tag.pth').to(device)
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name, param.size())
    net = net_train(0,net=net, train_loader=train_loader)


