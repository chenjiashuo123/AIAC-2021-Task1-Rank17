import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv1d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Inception(nn.Module):
    def __init__(self, feature_size=1536, hidden_size=384):
        super(Inception, self).__init__()

        self.conv1_1 = Conv1d(feature_size, hidden_size, kernel_size=1, padding=0)
        self.bn1_1 = nn.BatchNorm1d(hidden_size)
        self.conv1_2 = Conv1d(feature_size, hidden_size, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm1d(hidden_size)
        self.conv1_3 = Conv1d(feature_size, hidden_size, kernel_size=5, padding=2)
        self.bn1_3 = nn.BatchNorm1d(hidden_size)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv1_4 = Conv1d(feature_size, hidden_size, kernel_size=1, padding=0)
        self.bn1_4 = nn.BatchNorm1d(hidden_size)

        self.conv2_1 = Conv1d(feature_size, hidden_size, kernel_size=1, padding=0)
        self.bn2_1 = nn.BatchNorm1d(hidden_size)
        self.conv2_2 = Conv1d(feature_size, hidden_size, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm1d(hidden_size)
        self.conv2_3 = Conv1d(feature_size, hidden_size, kernel_size=5, padding=2)
        self.bn2_3 = nn.BatchNorm1d(hidden_size)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv2_4 = Conv1d(feature_size, hidden_size, kernel_size=1, padding=0)
        self.bn2_4 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.cat([
            F.relu(self.bn1_1(self.conv1_1(x))),
            F.relu(self.bn1_2(self.conv1_2(x))),
            F.relu(self.bn1_3(self.conv1_3(x))),
            F.relu(self.bn1_4(self.conv1_4(self.maxpool1(x)))),
            ], dim=1)
        x = torch.cat([
            F.relu(self.bn2_1(self.conv2_1(x))),
            F.relu(self.bn2_2(self.conv2_2(x))),
            F.relu(self.bn2_3(self.conv2_3(x))),
            F.relu(self.bn2_4(self.conv2_4(self.maxpool2(x)))),
            ], dim=1)
        vector = x.permute(0, 2, 1)
        return vector
        