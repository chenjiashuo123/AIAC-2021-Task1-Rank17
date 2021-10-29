from lichee.dataset.io_reader.io_reader_base import TFRecordReader
from lichee.utils.tfrecord.reader import read_single_record_with_spec_index
from lichee.utils.tfrecord import tfrecord_loader
from lichee.utils.tfrecord.tools import create_index
import concurrent.futures
from concurrent.futures import as_completed
from typing import List
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from torch.utils.data import Dataset
from transformers import  BertTokenizer, AutoTokenizer
import pandas as pd
import warnings
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")


class BaseDataset(Dataset):
    def __init__(self, data_path_list, desc_file,text_type='asr_text', training=True):
        self.data_path_list = data_path_list
        self.desc_file = desc_file
        self.description = self.get_desc()
        self.data_index_list = self.get_indexes()
        self.data_len = self.get_data_len()
        self.tfrecord_data_file_list = self.try_convert_to_tfrecord()
        self.training = training
        self.text_type = text_type
    
    def get_data_len(self, data_index_list):
        data_len = 0
        for data_index in data_index_list:
            data_len += len(data_index)
        return data_len
    def get_desc(self):
        reader_cls = TFRecordReader
        return reader_cls.get_desc(self.desc_file)

    def get_indexes(self):
        max_workers = min(8, len(self.data_path_list))
        reader_cls = TFRecordReader
        data_index_list = [None] * len(self.data_path_list)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            fs = {executor.submit(reader_cls.get_index, data_path, self.desc_file): i for i, data_path in
                    enumerate(self.data_path_list)}
            for future in as_completed(fs):
                data_index_list[fs[future]] = future.result()
        return data_index_list

    def get_nth_data_file(self, index):
        '''
        :param index: index of target item
        :return: item file index, and item start & end offset
        '''
        for i, data_index in enumerate(self.data_index_list):
            if index < len(data_index):
                break
            index -= len(data_index)
        start_offset = data_index[index]
        end_offset = data_index[index + 1] if index + 1 < len(data_index) else None
        return i, (start_offset, end_offset)
    def get_data_len(self):
        data_len = 0
        for data_index in self.data_index_list:
            data_len += len(data_index)
        return data_len

    def try_convert_to_tfrecord(self):
        tfrecord_data_file_list = []
        for data_path in self.data_path_list:
            reader_cls = TFRecordReader
            tfrecord_data_file_list.append(reader_cls.convert_to_tfrecord(data_path, self.desc_file))
        return tfrecord_data_file_list

    def __len__(self):
        return self.data_len

    def __getitem__(self,index):
        data_file_index, (start_offset, end_offset) = self.get_nth_data_file(index)
        tfrecord_data_file = self.tfrecord_data_file_list[data_file_index]
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset, self.description)
        vid = self.vid_process(row['id'])
        frames, frame_mask = self.frame_process(row['frame_feature'])
        frame = [frames, frame_mask]
        text = self.text_tokenizer_2(self.text_process(row['title']),self.text_process(row['asr_text']), max_length=64)
        if self.training:
            tag = self.tag_process(row['tag_id'])
            return vid, frame, text, tag
        else:
            return vid, frame, text

    def vid_process(self, id):
        return bytes(id).decode('utf-8')
    def frame_process(self, frames, num_segments=32):
        frame_feature = [np.frombuffer(bytes(x), dtype=np.float16).astype(np.float32) for x in frames]
        zero_frame = frame_feature[0] * 0.
        num_frames = len(frame_feature)

        if num_frames <= num_segments:
            padding_length = num_segments - num_frames
            res = frame_feature + [zero_frame] * (num_segments - num_frames)
            mask = [1] * (num_frames+1) + ([0] * padding_length)
        else:
            res = frame_feature[:, :num_segments]
            mask = [1] * (num_segments+1)
        return torch.tensor(np.c_[res]), torch.tensor(mask)
    def text_process(self, text):
        text = bytes(text).decode("utf-8")
        return text
    def tag_process(self, tag_id):
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
        tags = torch.LongTensor(multi_hot)
        return tags


    def category_process(self,  category_id, category_id_path='data/category_id.csv'):
        category_id_list=pd.read_csv(category_id_path, sep='\t', header=None)
        key = []
        for i in category_id_list[0]: 
            key.append(i)
    
        enc=LabelEncoder()   #获取一个LabelEncoder
        enc=enc.fit(key)  #训练LabelEncoder
        category_id=enc.transform(category_id)      
        category_id = torch.LongTensor(category_id)
        return torch.squeeze(category_id)
    
    def text_tokenizer(self,text_1, text_2=None, max_length=50):
        Tokenizer = AutoTokenizer.from_pretrained('data/macbert')
        if text_2 is not None:
            result = Tokenizer(text_1, text_2, padding="max_length", max_length=max_length, truncation=True)
        else:
            result = Tokenizer(text_1, padding="max_length", max_length=max_length, truncation=True)
        return torch.tensor(result['input_ids']), torch.tensor(result['token_type_ids']), torch.tensor(result['attention_mask'])

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


if __name__ == "__main__":

    # Tokenizer = BertTokenizer.from_pretrained('data/bert_base')
    # result = Tokenizer('测试1', None, padding="max_length", max_length=50, truncation=True)
    # print(result)
    data_path_list=['data/pairwise/pairwise.tfrecords']
    # # data_path_list=[ 'data/test_a/test_a.tfrecords']
    # desc_file = 'data/desc.json'

    # dataset = BaseDataset(data_path_list, desc_file, text_type='com_text', training=True)
    # train_loader = DataLoader(dataset, batch_size=2,
    #                 shuffle=True)
    # for i, (vid, frame, text, tag, category) in enumerate(train_loader):
    #     # continue
    #     total = 0
    #     correct = 0
    #     # net = TAGNET().cuda()
    #     # out = net(text, frame, tag)
    #     # prob = out[1]
    #     # total += 10000 * 256
    #     # correct += ((prob > 0.5) == tag).sum().item()
    #     # print(vid)
    #     # print(category)
    #     if i == 2:
    #         break
    # label_list = set([2,3,4,5,6,1])
    # mlb = MultiLabelBinarizer()
    # mlb.fit([label_list])
    # one_hot = LabelEncoder()
    # one_hot.fit([label_list])
    # a = mlb.transform([(1, 2), (3,4),(5,)])
    # b = one_hot.transform([(1, 2), (3,4),(5,)])
    # print(a)

