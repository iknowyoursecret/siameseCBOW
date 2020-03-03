import torch as t
import torch.nn as nn


class Average(nn.Module):  # 自定义层:求平均 加和平均
    def __init__(self):
        super(Average, self).__init__()

    def forward(self, s):
        ave = t.mean(s, 1)
        return ave
