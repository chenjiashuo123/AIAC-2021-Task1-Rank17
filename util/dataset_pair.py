from lichee.dataset.io_reader.io_reader_base import BaseIOReader, TFRecordReader
from lichee.utils import common
from lichee.utils.tfrecord.reader import read_single_record_with_spec_index
from lichee.utils.tfrecord import tfrecord_loader
from lichee.utils.tfrecord.tools import create_index
import concurrent.futures
from abc import ABCMeta
from concurrent.futures import as_completed
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from transformers import  AutoTokenizer
import warnings
from sklearn.metrics import f1_score
from lichee.utils.tfrecord import example_pb2
from lichee.utils.tfrecord.reader import tfrecord_iterator
import pandas as pd
warnings.filterwarnings("ignore")


class BaseDataset(Dataset):
    def __init__(self, df_path, data_path,text_type='title'):
        self.df_path = df_path

        self.text_type = text_type
        self.data_path = data_path

    def __len__(self):
        return len(self.df_path)

    def __getitem__(self,index):
        datas = self.df_path.iloc[index, 0].split('\t')
        vid_1 = datas[0]
        vid_2 = datas[1]
        label = float(datas[2])
         
        v_content_1 = self.read_record(vid_1)
        v_content_2 = self.read_record(vid_2)
        return (vid_1, v_content_1), (vid_2, v_content_2), label

    def read_record(self, target):
        frame_path = self.data_path + 'frame_feature/' + target + '.npy'
        title_path = self.data_path + 'title/' + target + '.txt'
        asr_text_path = self.data_path + 'asr_text/' + target + '.txt'
        frame_feature = self.frame_process(frame_path)
        if self.text_type == 'asr_text':
            text = self.text_tokenizer(asr_text_path)
        elif self.text_type == 'title':
            text =  self.text_tokenizer(title_path)
        elif self.text_type == 'com_text':
            text = self.text_tokenizer(title_path, asr_text_path, max_length=64)
        return frame_feature, text
        
    def frame_process(self, frames_path, num_segments=32):
        frame_feature = np.load(frames_path)
        zero_frame = frame_feature[0] * 0.
        num_frames = frame_feature.shape[0]
        dim = frame_feature.shape[1]

        if num_frames <= num_segments:
            padding_length = num_segments - num_frames
            fillarray = np.zeros((padding_length, dim))
            res = np.concatenate((frame_feature, fillarray), axis=0)
            mask = [1] * num_frames + ([0] * padding_length)
        else:
            res = frame_feature[:, :num_segments]
            mask = [1] * num_segments
        return torch.tensor(np.c_[res], dtype=torch.float32), torch.tensor(mask)
    def tag_process(self, tag_id_path):
        tag_id = np.loadtxt(tag_id_path,delimiter=',', dtype=np.int32)
        tag_id = np.atleast_1d(tag_id) 
        mlb = MultiLabelBinarizer()
        selected_tags = set()
        tag_file = 'data/tag_list.txt'
        with open(tag_file, encoding='utf-8') as fh:
            for line in fh:
                fields = line.strip().split('\t')
                selected_tags.add(int(fields[0]))
        mlb.fit([selected_tags])
        tags = [t for t in tag_id if t in selected_tags]
        multi_hot = mlb.transform([tags])[0]
        label = torch.LongTensor(multi_hot)
        return label
    def text_tokenizer(self,text_1_path, text_2_path=None,max_length=32):
        text_2 = None
        with open(text_1_path, "r") as f:  
            text_1 = f.read()
        if text_2_path is not None:
            with open(text_2_path, "r") as f:  
                text_2 = f.read()
        return self.text_tokenizer_2(text_1, text_2, max_length)

    def text_tokenizer_2(self,text,text_2=None, max_length=50):
        Tokenizer = AutoTokenizer.from_pretrained('data/bert_base')
        PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'
        token = Tokenizer.tokenize(text)
        token_1 = [CLS] + token + [SEP]
        seq_len = len(token)
        mask = []
        token_2 = []
        token = token_1
        if text_2 is not None:
            token_2 = Tokenizer.tokenize(text_2)
            token = token_1 + token_2
        token_ids = Tokenizer.convert_tokens_to_ids(token)
        if len(token) < max_length:
            mask = [1] * len(token_ids) + [0] * (max_length - len(token))
            token_ids += ([0] * (max_length - len(token)))
            token_type_ids = [0]*len(token_1) + [1]*len(token_2) + [0]*(max_length - len(token))
        else:
            mask = [1] * max_length
            token_ids = token_ids[:max_length]
            token_type_ids = [0]*len(token_1) + [1]*len(token_2)
            token_type_ids = token_type_ids[:max_length]
        return torch.LongTensor(token_ids), torch.LongTensor(token_type_ids), torch.LongTensor(mask)

def tag_process(tag_id_path):
    tag_id = np.loadtxt(tag_id_path,delimiter=',', dtype=np.int32)
    tag_id = np.atleast_1d(tag_id) 
    
    mlb = MultiLabelBinarizer()
    selected_tags = set()
    tag_file = 'data/tag_list.txt'
    with open(tag_file, encoding='utf-8') as fh:
        for line in fh:
            fields = line.strip().split('\t')
            selected_tags.add(int(fields[0]))
    mlb.fit([selected_tags])
    tags = [t for t in tag_id if t in selected_tags]
    multi_hot = mlb.transform([tags])[0]
    label = torch.LongTensor(multi_hot)
    return label

def category_process(category_id_path):

    category_id=pd.read_csv(category_id_path, sep='\t', header=None)
    key = []
    for i in category_id[0]: #“number”用作键
        key.append(i)
 
    tag_id = np.loadtxt('data/finetune/category_id/3326436038933882.txt',delimiter=',', dtype=np.int32)
    tag_id = np.atleast_1d(tag_id) 
    enc=LabelEncoder()   #获取一个LabelEncoder
    enc=enc.fit(key)  #训练LabelEncoder
    data=enc.transform(tag_id)       #使用训练好的LabelEncoder对原数据进行编码
    label = torch.LongTensor(data)
 
    print(label)    #输出编码后的数据
    return label
 
    
    # enc = MultiLabelBinarizer()
    # selected_tags = set()
    # tag_file = 'data/tag_list.txt'
    # with open(tag_file, encoding='utf-8') as fh:
    #     for line in fh:
    #         fields = line.strip().split('\t')
    #         selected_tags.add(int(fields[0]))
    # mlb.fit([selected_tags])
    # tags = [t for t in tag_id if t in selected_tags]
    # multi_hot = mlb.transform([tags])[0]
    # label = torch.LongTensor(multi_hot)
    # return label

# category_process('data/category_id.csv')




# if __name__ == "__main__":
#     data_path_list='data/finetune/'
#     desc_file = 'data/desc.json'

#     df_train = pd.read_csv('sort_label.tsv', header=None)
#     dataset = BaseDataset(df_train, data_path_list)
#     print(len(dataset))
#     train_loader = DataLoader(dataset, batch_size=64,num_workers=1,shuffle=True)
    
#     for i, (v_content_1, v_content_2, label)  in enumerate(train_loader):
#         vid_1 = v_content_1[0]
#         frame_feature_1 = v_content_1[1][0]
#         tag_id_1 = v_content_1[1][1]
#         tag_id_2 = v_content_2[1][1]
#         text_1 = v_content_1[1][2]
#         print(vid_1)
#         print(tag_id_1)
#         print(tag_id_2)



