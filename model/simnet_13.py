import torch
import numpy as np
import transformers
import torch.nn as nn
import torch.nn.functional as F
from model.video_model import VIDEONET
from model.text_model_focal_2 import TEXTNET
from model.channel_attention import SELayer, Cross_SE, Cross_SE_BN, Cross_SE_BN_DROP



class SIMNET(nn.Module):
    def __init__(self, device, ckpt_text=None, ckpt_video=None):
        super(SIMNET, self).__init__()

        self.text_model = TEXTNET()
        if ckpt_text is not None:
            self.text_model.load_state_dict(torch.load(ckpt_text, map_location=device))
        for name, param in self.text_model.named_parameters():
            param.requires_grad = False
        self.video_model = VIDEONET()
        if ckpt_video is not None:
            self.video_model.load_state_dict(torch.load(ckpt_video, map_location=device))
        for name, param in self.video_model.named_parameters():
            param.requires_grad = False

        self.video_embed = nn.Linear(1536, 256)
        self.text_embed = nn.Linear(768, 256)
        self.fusion_gate = Cross_SE_BN(512)

    def forward(self, text_ids, frames, labels=None):
        _, video_embeddings = self.video_model.extract_features(frames)
        _, text_embeddings = self.text_model.extract_features(text_ids)
        text_embeddings = self.text_embed(text_embeddings)
        video_embeddings = self.video_embed(video_embeddings)
        embeddings = self.fusion_gate(video_embeddings, text_embeddings)
        return embeddings

if __name__ == "__main__":
    from torch.autograd import Variable
    video = Variable(torch.ones(4, 32,1536))
    net = TAGNET()
    out = net(video)
    print(out[0].shape)

