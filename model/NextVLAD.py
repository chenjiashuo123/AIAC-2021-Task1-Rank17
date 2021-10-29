import torch
import torch.nn as nn
import torch.nn.functional as F





class NeXtVLAD(nn.Module):

    def __init__(self, feature_size=1536, output_size=1024, expansion_size=2, cluster_size=64, num_groups=8, dropout_prob=0.2):
        super(NeXtVLAD, self).__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion_size
        self.cluster_size = cluster_size
        self.groups = num_groups
        self.drop_rate = dropout_prob

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)
        # self.apply(weights_init_kaiming)

    def forward(self, inputs):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad


if __name__ == '__main__':
    from torch.autograd import Variable
    config = '../pretrain_model/bert_wwm_ext/config.json'
    path = '../pretrain_model/bert_wwm_ext/chinese_wwm_ext.bin'
    video = Variable(torch.ones(4, 120,128))
    # text = Variable(torch.ones((4,128), dtype=torch.int64))
    net = NeXtVLAD()
    out = net(video)
    print(out.shape)
