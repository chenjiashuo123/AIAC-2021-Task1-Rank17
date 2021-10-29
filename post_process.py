import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, paired_distances
import torch
import torch.nn.functional as F
from transformers import  BertTokenizer
import torch.nn as nn
if __name__=='__main__':
    import json
    with open('result_multi_bert_1_all.json', 'r', encoding = 'utf-8') as f:
        a = json.load(f)
    with open('result_simnet_17_without_drop_all.json', 'r', encoding = 'utf-8') as f:
        b = json.load(f)
    res_a = np.array([item for item in a.values()])
    res_b = np.array([item for item in b.values()])
    list_vid = [item for item in b.keys()]
    res = res_a + res_b
    embedding = torch.tensor(res)
    embedding = F.normalize(embedding, p=2, dim=1)
    embedding = embedding.numpy().astype(np.float16).tolist()
    output_res = dict(zip(list_vid, embedding))
    with open('result/result.json', 'w') as f:
        json.dump(output_res, f)