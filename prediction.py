import torch as t

from config import opt
from data.dataset import loadData
from model.siameseCBOW import SiasemeCBOW
from preprocess.generateWordDic import generate_word_dic
from preprocess.sentenceprocess import data2num_list

if __name__ == '__main__':

    # 用于训练的数据
    dat = loadData(opt.data_path)
    # 数据的长度，即句子数
    pass
    # 单词字典
    _, word_dic = generate_word_dic(dat)
    print("请输入主句：")
    main_str = input()
    # str_ls = []
    print("测试句子数目")
    other_str_num = eval(input())
    print("请输入测试句：")
    other_str = [input() for _ in range(other_str_num)]
    str_ls = [main_str] + other_str
    # 转tensor
    sent_length = opt.sent_length
    test_tensor = data2num_list(str_ls, word_dic, sent_length)
    input = t.tensor(test_tensor)

    # 词向量的长度
    input_dim = len(word_dic)
    # 词嵌入的输输出维度
    output_dim = opt.output_dim

    sent_length = opt.sent_length

    n_pos = opt.n_pos
    n_neg = opt.n_neg
    # 模型实例化
    model = SiasemeCBOW(input_dim, output_dim, n_pos, n_neg)
    # 如果模型有之前训练的参数 就用之前的参数，没有的话重头训练
    # path = None
    # path = r'output/SiasemeCBOW0219_14_46_43.pth'
    if opt.model_param_path is not None:
        model.load(opt.model_param_path)
    else:
        print("目前还未有训练出参数集")

    _, predict = model(input)
    max_value, idx = t.max(predict, dim=1)
    print("最相似句子为：第%d个句子,概率为%f" % (idx + 1, max_value))
