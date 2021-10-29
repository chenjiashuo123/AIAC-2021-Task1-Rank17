import json

import numpy as np
from lichee.utils.tfrecord import example_pb2
from lichee.utils.tfrecord.reader import tfrecord_iterator
import torch
import pandas as pd

import matplotlib.pyplot as plt
import os

def gen_finetune_data():
    record_iterator = tfrecord_iterator('data/pairwise/pairwise.tfrecords')
    dir_1 = 'data/finetune/frame_feature/'
    dir_2 = 'data/finetune/tag_id/'
    dir_3 = 'data/finetune/asr_text/'
    dir_4 = 'data/finetune/title/'
    dir_5 = 'data/finetune/category_id/'
    if not os.path.isdir(dir_1):  
        os.makedirs(dir_1)
    if not os.path.isdir(dir_2): 
        os.makedirs(dir_2)
    if not os.path.isdir(dir_3):  
        os.makedirs(dir_3)
    if not os.path.isdir(dir_4):  
        os.makedirs(dir_4)
    if not os.path.isdir(dir_5):
        os.makedirs(dir_5)
    for i, record in enumerate(record_iterator):
        example = example_pb2.Example()
        example.ParseFromString(record)
        all_keys = list(example.features.feature.keys())

        field = example.features.feature['id'].ListFields()[0]
        value = field[1].value
        vid = bytes(np.frombuffer(value[0], dtype=np.uint8)).decode()


        field = example.features.feature['frame_feature'].ListFields()[0]
        value = field[1].value
        frame_feature = [np.frombuffer(bytes(x), dtype=np.float16).astype(np.float32) for x in value]
        frame_feature = np.c_[frame_feature]
        save_name = dir_1 + vid + '.npy'
        np.save(save_name, frame_feature)


        field = example.features.feature['tag_id'].ListFields()[0]
        value = field[1].value
        tag_id = np.array(value, dtype=np.int32)
        tag_id = ','.join(str(i) for i in tag_id)
        save_name = dir_2 + vid + '.txt'
        with open(save_name,"w") as f:
            f.write(tag_id)


        field = example.features.feature['category_id'].ListFields()[0]
        value = field[1].value
        category_id = np.array(value, dtype=np.int32)
        category_id = ','.join(str(i) for i in category_id)
        save_name = dir_5 + vid + '.txt'
        with open(save_name,"w") as f:
            f.write(category_id)

        field = example.features.feature['title'].ListFields()[0]
        value = field[1].value
        title = bytes(np.frombuffer(value[0], dtype=np.uint8)).decode()
        save_name = dir_4 + vid + '.txt'
        with open(save_name,"w") as f:
            f.write(title)


        field = example.features.feature['asr_text'].ListFields()[0]
        value = field[1].value
        asr_text = bytes(np.frombuffer(value[0], dtype=np.uint8)).decode()
        save_name = dir_3 + vid + '.txt'
        with open(save_name,"w") as f:
            f.write(asr_text)

if __name__ == '__main__':
    gen_finetune_data()