import os
import time

import torch as t
import torch.nn as nn

from layer.averageLayer import Average
from layer.cosineLayer import CosineLayer
from layer.embeddingLayer import Embedding


class BaseModule(nn.Module):
    """
    基础模型，用于实现部分通用功能
    BaseModule:主要作用是读取和保存参数
    """

    def __init__(self):
        super(BaseModule, self).__init__()
        self.model_name = self.__class__.__name__  # str(type(self))  # 默认名字#

    def load(self, path):
        """
        可加载指定路径的模型
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        # print("1")
        if name is None:
            # print(os.getcwd())
            prefix = os.path.join(os.getcwd(), 'output')
            # prefix = os.getcwd() + '/ouput/' + self.model_name + '_'
            name = time.strftime(self.model_name + '%m%d_%H_%M_%S.pth')
            name = os.path.join(prefix, name)
            # print(type(name))
            # name = r"D:\paperAndCode\PycharmProjects\siameseCBOW\output\BaseModule"
        t.save(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_decay):
        return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class SiasemeCBOW(BaseModule):
    """
    input_dim:字典的长度，字的个数
    ouput_dim:Embedding dimension 嵌入的维度
    seq_length = 64 # Sentence length 长了截取 短了补充
    input_length:
    n_positive:Number of positice sample
    n_negative:Number of negative sample
    """

    def __init__(self, input_dim, output_dim=100, n_positive=2, n_negative=2):
        super(SiasemeCBOW, self).__init__()
        self.embeds = Embedding(input_dim, output_dim)  # .from_pretrained(load_weight())
        self.drop = nn.Dropout()
        self.ave = Average()
        # self.relu = nn.ReLU
        self.cosine = CosineLayer(n_positive, n_negative)
        self.lossCEL = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        w = self.embeds(inputs)
        # w = self.drop(w)
        # w = t.FloatTensor(w)

        ave_s = self.ave(w)
        # print(ave_s)
        cos = self.cosine(ave_s)
        # loss = self.lossCEL(n_pos)
        predict = self.softmax(cos)
        # print(predict)
        # loss=self.lossCEL(cos)
        return cos, predict


if __name__ == '__main__':
    model = BaseModule()
    model.save()
