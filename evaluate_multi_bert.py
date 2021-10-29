# coding: UTF-8
import time
from numpy.lib.function_base import delete
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
import os
import json
import scipy
import torch
from model.multi_bert_simnet_1 import SIMNET
from util.dataset import BaseDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_checkpoint(checkpoint_file, test_dataloader, to_save_file):
    model = SIMNET(device = device)
    model.load_state_dict(torch.load(checkpoint_file, map_location=device))
    model = model.to(device)
    model.eval()
    id_list = []
    embedding_list = []
    with torch.no_grad():
        for vid, frame, text in test_dataloader:
            text = text
            input_id = text[0].to(device)
            token_type_ids = text[1].to(device)
            attention_mask = text[2].to(device)

            video_feature = frame[0].to(device)
            video_mask = frame[1].to(device)
            embedding = model((input_id, token_type_ids, attention_mask), (video_feature, video_mask))
            id_list.extend(list(vid))
            # embedding = F.normalize(embedding, p=2, dim=1)
            embedding = embedding.detach().cpu().numpy()
            embedding_list.extend(embedding.astype(np.float16).tolist())
            output_res = dict(zip(id_list, embedding_list))
    with open(to_save_file, 'w') as f:
        json.dump(output_res, f)

if __name__ == '__main__':
    # 单模型预测
    checkpoint_file = 'checkpoint/multi_bert_unfrozen_1_unsup_all_epoch_6.pth'
    test_data_path_list=['data/test_b/test_b.tfrecords']
    desc_file = 'data/desc.json'
    to_save_file = 'result_multi_bert_all.json'
    test_dataset = BaseDataset(test_data_path_list, desc_file, text_type='title', training=False)
    test_loader = DataLoader(test_dataset, batch_size=256,num_workers=8,pin_memory=True, shuffle=False)
    evaluate_checkpoint(checkpoint_file, test_loader, to_save_file)