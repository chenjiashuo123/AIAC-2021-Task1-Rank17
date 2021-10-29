import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer_cls import Transformer
from model.losses import cls_loss_func
from model.losses import BinaryFocalLoss


class VIDEONET(nn.Module):
    def __init__(self):
        super(VIDEONET, self).__init__()

        self.video_model = Transformer(seq_len=32, dim=1536, depth=6, heads=12, dim_head=256, mlp_dim=3072, dropout=0.1, emb_dropout=0.1)
        self.cls =  nn.Linear(1536, 10000)
        self.loss_fn = BinaryFocalLoss()

    def extract_features(self, frames):
        video_feature_trans = self.video_model(frames[0], frames[1])
        embeddings = (video_feature_trans * frames[1].unsqueeze(-1)).sum(1)/(frames[1].sum(-1)+1e-10).unsqueeze(-1)
        return video_feature_trans, embeddings
    
    def forward(self, frames, tag_labels=None):
        _,embeddings = self.extract_features(frames)
        tag_probs = torch.sigmoid(self.cls(embeddings))
        if tag_labels is not None:
            loss_tag = self.loss_fn(tag_probs, tag_labels.float())
            loss = loss_tag*0.001
            return loss, tag_probs, embeddings
        else:
            return tag_probs, embeddings  

