import torch
import numpy as np
import transformers
import torch.nn as nn
import torch.nn.functional as F
from model.multi_bert import MULTIBERT
from model.channel_attention import SELayer, Cross_SE, Cross_SE_BN, SE_BN



class SIMNET(nn.Module):
    def __init__(self, device, ckpt=None):
        super(SIMNET, self).__init__()

        self.bert = MULTIBERT()
        if ckpt is not None:
            self.bert.load_state_dict(torch.load(ckpt, map_location=device))
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
        for name, param in self.bert.multi_bert.named_parameters():
            param.requires_grad = True
        self.fusion_gate = SE_BN(1536)
        self.embed = nn.Linear(1536, 256)


    def forward(self, text_ids, frames, labels=None):
        _, _, cat_embeddings = self.bert.extract_features(text_ids, frames)
        embeddings = self.fusion_gate(cat_embeddings)
        embeddings = self.embed(embeddings)
        return embeddings


