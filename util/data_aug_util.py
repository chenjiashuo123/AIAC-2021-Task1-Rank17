from pandas.core.frame import DataFrame
import pandas as pd
import json

import numpy as np
import torch
import pandas as pd

import matplotlib.pyplot as plt
import os
import itertools


def aug_sim_one_video_pair():
    pair_df_1 = pd.read_csv('finetune_train.tsv', sep='\t', header=None)
    pair_df_1.columns = ['vid_1','vid_2', 'label']
    sim_1_video_pair = pair_df_1[pair_df_1['label'] == 1]
    list_one_vids = sim_1_video_pair['vid_1'].values.tolist()
    list_one_vids += sim_1_video_pair['vid_2'].values.tolist()
    list_one_vids=list(set(list_one_vids))
    
    new_pair_list = []
    i = 0 
    for vid in list_one_vids:
    # vid = 1003139095736090030
        df_1 = sim_1_video_pair[sim_1_video_pair['vid_1'] == int(vid)]
        df_2 = sim_1_video_pair[sim_1_video_pair['vid_2'] == int(vid)]
        pair_vid = df_1['vid_2'].values.tolist()
        pair_vid += df_2['vid_1'].values.tolist()
        pair_vid=list(set(pair_vid))
        new_pair = list(itertools.combinations(pair_vid, 2))
        if len(new_pair) > 0:
            new_pair_list += new_pair
    data=DataFrame(columns=['vid_1','vid_2'], data = new_pair_list)#这时候是以行为标准写入的
    data['label'] = 1
    pair_train_aug = pd.concat([pair_df_1,data],axis=0)
    pair_train_aug = pair_train_aug.drop_duplicates()
    pair_train_aug.to_csv('finetune_train_aug_1.csv',index=False, header=None,  sep='\t')

aug_sim_one_video_pair()

