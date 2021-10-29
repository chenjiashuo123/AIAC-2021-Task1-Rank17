import torch
import torch.nn as nn
from model.losses import cls_loss_func, BinaryFocalLoss
from transformers import AutoModel




class TEXTNET(nn.Module):
    def __init__(self):
        super(TEXTNET, self).__init__()
        self.text_model = AutoModel.from_pretrained('data/bert_base')
        for param in self.text_model.parameters():
            param.requires_grad = True
        self.cls = nn.Linear(768, 10000)
        self.loss_fn = BinaryFocalLoss()


    def extract_features(self, text):
        text_feature = self.text_model(input_ids=text[0], token_type_ids=text[1], attention_mask=text[2])
        text_feature_sequence = text_feature[0]
        embeddings = text_feature_sequence[:,0]
        return text_feature_sequence, embeddings
    
    def forward(self, text, tag_labels=None):
        _, embeddings = self.extract_features(text)
        tag_probs = torch.sigmoid(self.cls(torch.relu(embeddings)))
        if tag_labels is not None:
            loss = self.loss_fn(tag_probs, tag_labels.float())
            loss = loss * 0.001
            return loss, tag_probs, embeddings
        else:
            return tag_probs, embeddings  

