import torch as t
import torch.nn as nn
import torch.nn.functional as F


class CosineLayer(nn.Module):  # 求余弦相似度 这一层需要修改
    def __init__(self, num_postive, num_negtive):
        super(CosineLayer, self).__init__()
        self.n_postive = num_postive
        self.n_negtive = num_negtive
        # self.normlize = F.normalize()

    def forward(self, ave_result):
        ave_result = F.normalize(ave_result)

        cos_main = ave_result[0].view((1, len(ave_result[0])))
        # cos_main = self.normlize(cos_main)
        cosine = t.cosine_similarity(cos_main, ave_result, dim=1)
        # print(cosine)
        # print(cosine.shape)
        return cosine[1:].view((1, ave_result.shape[0]-1))
        # return cosine[1:1 + self.n_postive].view((1, self.n_negtive))
