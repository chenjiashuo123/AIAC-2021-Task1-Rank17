import torch
from torch import nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=64, multiply=True):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_avg = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
                )
        self.multiply = multiply
    def forward(self, x):
        b, c, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_avg = self.fc_avg(y_avg).view(b, c, 1)
        out_avg = x*y_avg
        out = out_avg 
        return out

class Cross_SE(nn.Module):
    def __init__(self, channel):
        super(Cross_SE, self).__init__()
        self.fc_avg = nn.Sequential(
                nn.Linear(channel, channel),
                nn.ReLU(inplace=True),
                nn.Sigmoid()
                )
    def forward(self, video, text):
        b, c= video.size()
        x = torch.cat((video, text), -1)
        y_avg = self.fc_avg(x)
        video_se = y_avg[:,:c]
        text_se = y_avg[:,c:]
        out = video_se*video + text_se*text
        return out


class SE_BN(nn.Module):
    def __init__(self, channel):
        super(SE_BN, self).__init__()
        self.fc_avg = nn.Sequential(
                nn.Linear(channel, channel),
                nn.BatchNorm1d(channel),
                nn.Sigmoid()
                )
    def forward(self, input):
        y_avg = self.fc_avg(input)
        out = input*y_avg
        return out

class Cross_SE_BN(nn.Module):
    def __init__(self, channel):
        super(Cross_SE_BN, self).__init__()
        self.fc_avg = nn.Sequential(
                nn.Linear(channel, channel),
                nn.BatchNorm1d(channel),
                nn.Sigmoid()
                )
    def forward(self, video, text):
        b, c= video.size()
        x = torch.cat((video, text), -1)
        y_avg = self.fc_avg(x)
        video_se = y_avg[:,:c]
        text_se = y_avg[:,c:]
        out = video_se*video + text_se*text
        return out


class Cross_SE_BN_DROP(nn.Module):
    def __init__(self, channel):
        super(Cross_SE_BN_DROP, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.fc_avg = nn.Sequential(
                nn.Linear(channel, channel),
                nn.BatchNorm1d(channel),
                nn.Sigmoid()
                )
    def forward(self, video, text):
        _, c= video.size()
        x = torch.cat((video, text), -1)
        x = self.dropout(x)
        y_avg = self.fc_avg(x)
        video_se = y_avg[:,:c]
        text_se = y_avg[:,c:]
        out = video_se*video + text_se*text
        return out

class Cross_SE_BN_DROP_v2(nn.Module):
    def __init__(self, channel):
        super(Cross_SE_BN_DROP_v2, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.fc_avg = nn.Sequential(
                nn.Linear(channel, channel),
                nn.BatchNorm1d(channel),
                nn.Sigmoid()
                )
    def forward(self, video, text):
        _, c= video.size()
        x = torch.cat((video, text), -1)
        x = self.dropout(x)
        y_avg = self.fc_avg(x)
        video_se = y_avg[:,:c]
        text_se = y_avg[:,c:]
        video_emded = video_se*video 
        text_emded = text_se*text
        return video_emded, text_emded

