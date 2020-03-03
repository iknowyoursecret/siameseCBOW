from torch.utils import data
import os


class loadData(data.Dataset):

    def __init__(self, file_name):  # r'C:\Users\xiaopeng`s lap\Desktop\诡秘之主.txt'
        self.corpus = []  # 语料
        if os.path.isfile(file_name):
            with open(file_name, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line != '\n':
                        self.corpus.append(line.strip())
        elif os.path.isdir(file_name):
            files = os.listdir(file_name)  # 得到文件夹下的所有文件名称
            for file in files:  # 遍历文件夹
                if os.path.isfile(file):  # 判断是否是文件夹，不是文件夹才打开
                    with open(os.path.join(file_name, file), 'r') as f:  # 打开文件
                        lines = f.readlines()
                        for line in lines:
                            if line != '\n':
                                self.corpus.append(line.strip())

    # 返回df的长度
    def __len__(self):
        return len(self.corpus)

    # 获取第idx+1列的数据
    def __getitem__(self, idx):
        return self.corpus[idx]

