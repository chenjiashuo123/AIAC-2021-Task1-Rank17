import torch
import torch.nn as nn
from model.until_module import LayerNorm
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from model.losses import BinaryFocalLoss

class BertOnlyMLMHead(nn.Module):
    def __init__(self):
        super(BertOnlyMLMHead, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = LayerNorm(768, eps=1e-12)

        self.decoder = nn.Linear(768,21128,bias=False)
        self.bias = nn.Parameter(torch.zeros(21128))
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class VisualEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, vocab_size=1536, hidden_size=768, max_position_embeddings=32, hidden_dropout_prob=0.1):
        super(VisualEmbeddings, self).__init__()

        self.word_embeddings = nn.Linear(vocab_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_embeddings):
        embeddings = self.word_embeddings(input_embeddings)
        embeddings = self.LayerNorm(embeddings)
        return embeddings

class MULTIBERT(nn.Module):
    def __init__(self, do_pretrain=True, pretrained_text_model='data/bert_base'):
        super(MULTIBERT, self).__init__()
        self.do_pretrain = do_pretrain
        self.text_model = BertModel.from_pretrained(pretrained_text_model)
        config = BertConfig().from_pretrained(pretrained_text_model)
        self.vocab_size = config.vocab_size
        self.visual_embeddings = nn.Linear(1536, 768)
        self.text_ln = nn.LayerNorm(768)
        config.type_vocab_size = 2
        config.num_attention_heads = 8
        config.num_hidden_layers = 6
        self.multi_bert = BertModel(config)
        self.cls = nn.Linear(1536, 10000)
        self.loss_fn = BinaryFocalLoss()

    def extract_features(self, text, frames):
        B, L = text[0].shape
        video_featrue = self.visual_embeddings(frames[0])
        text_feature = self.text_model(input_ids=text[0], attention_mask=text[2])
        text_feature = text_feature[0]
        text_feature = self.text_ln(text_feature)

        token_type_ids = text[1].long()
        pad_ids = torch.ones_like(token_type_ids) * 1
        token_type_ids = torch.cat([token_type_ids, pad_ids], dim=-1)
        cat_mask = torch.cat([text[2], frames[1]], dim=-1)
        cat_feature = torch.cat([text_feature, video_featrue], dim=1)

        cat_feature_all = self.multi_bert(inputs_embeds=cat_feature, token_type_ids=token_type_ids, attention_mask=cat_mask, output_hidden_states=True)
        cat_feature = cat_feature_all[0]
        video_feature = cat_feature[:, L:, :]
        text_feature_pool = cat_feature[:, 0]
        video_feature_pool = (video_feature * frames[1].unsqueeze(-1)).sum(1)/(frames[1].sum(-1)+1e-10).unsqueeze(-1)
        embeddings = torch.cat((text_feature_pool, video_feature_pool), -1)
        return text_feature, video_feature, embeddings
    def forward(self, text, frames, tag_labels=None):
        _, _, embeddings = self.extract_features(text, frames)
        tag_probs = torch.sigmoid(self.cls(embeddings))
        if tag_labels is not None:
            loss = self.loss_fn(tag_probs, tag_labels.float())
            loss = loss * 0.001
            return loss, tag_probs, embeddings
        else:
            return tag_probs, embeddings  

        



