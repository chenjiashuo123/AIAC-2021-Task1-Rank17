import torch
import torch.nn as nn
import torch.nn.functional as F


video = torch.ones(4,8,256)
print(video)
fold = 1
out = torch.zeros_like(video)
out[:, :-1, :fold] = video[:, 1:, :fold]  # shift left
out[:, 1:, fold: 2 * fold] = video[:, :-1, fold: 2 * fold]  # shift right
out[:, :, 2 * fold:] = video[:, :, 2 * fold:]  # not shift

print(out[0,:,:])