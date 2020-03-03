import numpy as np
import torch as t
from torch import nn, optim

from config import opt
from data.dataset import loadData
from model.siameseCBOW import SiasemeCBOW
from preprocess.generateWordDic import generate_word_dic
from preprocess.getPosData import get_postive_data_ls
from preprocess.sentenceprocess import data2num_list

if __name__ == '__main__':
    '''
    STEP 1: 数据导入
    '''
    # 用于训练的数据
    dat = loadData(opt.data_path)
    # 数据的长度，即句子数
    pass
    # 单词字典
    _, word_dic = generate_word_dic(dat)

    # 词向量的长度
    input_dim = len(word_dic)
    # 词嵌入的输输出维度
    output_dim = opt.output_dim
    # 用于截取或扩充的句子长度
    sent_length = opt.sent_length

    n_pos = opt.n_pos
    n_neg = opt.n_neg

    input_data = data2num_list(dat, word_dic, sent_length)
    # print(input_data)
    # 句子数量
    sent_num = len(input_data)
    print("has %d sentences will be used to train" % sent_num)
    # 转换成张量
    # input_data = t.tensor(input_data_ls)

    # print(input_data.type())  # 类型正确
    # print(input_data[10])  # 数据正确
    # print(len(input_data))

    '''
    STEP 2: 实例化模型
    '''
    model = SiasemeCBOW(input_dim, output_dim, n_pos, n_neg)
    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(device)
    if t.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(t.cuda.device_count()))
        model = nn.DataParallel(model)
    model = model.to(device)
    # 如果模型有之前训练的参数 就用之前的参数，没有的话重头训练
    if opt.model_param_path is not None:
        print("loaded model parameters")
        model.load(opt.model_param_path)
    # device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    # model.to(device)

    '''
    STEP 3: 实例化损失函数
    '''
    # 交叉熵损失函数
    # lossCEL = nn.CrossEntropyLoss(reduce=True,size_average=False,weight=embeds.weight)
    lossCEL = nn.CrossEntropyLoss()
    # lossCus = CustomLoss()
    # lossMSE = nn.MSELoss()  # 均方损失函数
    # lossMSE = nn.MultiLabelSoftMarginLoss()
    # lossNLLL = nn.NLLLoss()
    '''
    STEP 4:  实例化优化器
    '''
    # 这里能设置成参数的只有embed的结果吧
    # optim = optim.adam()
    learning_rate = opt.learning_rate
    # optimizer = t.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    '''
    STEP 5:  模型训练
    '''
    for epoch in range(opt.epochs):
        running_loss = 0.0
        for idx, data in enumerate(input_data, 0):
            if idx == 0:
                continue
            if idx > len(input_data) - n_pos - n_neg:
                break
            # 主句
            main_input = input_data[idx]
            # print("main_input is already loaded")
            pos_ls = get_postive_data_ls(idx, n_pos)
            # 正例集
            pos_inputs = [input_data[pos_idx] for pos_idx in pos_ls]
            # print("pos_inputs is already loaded")
            ls = pos_ls + [idx]
            # print("ls:{}".format(ls))
            # 反例集
            neg_inputs = []
            while len(neg_inputs) < n_neg:
                temp_num = np.random.randint(0, sent_num)
                if temp_num not in ls:
                    neg_inputs.append(input_data[temp_num])
            # print("neg_inputs is already loaded")
            # print(neg_inputs)
            inputs = [main_input] + pos_inputs + neg_inputs
            inputs = t.tensor(inputs).to(device)
            # labels = t.tensor([1 / n_pos] * n_pos + [0] * n_neg).float()
            # labels = t.tensor([1] * n_pos + [0] * (n_neg)).long()
            target = t.tensor([0]).to(device)
            # Clear gradients w.r.t. parameters 参数梯度置零
            optimizer.zero_grad()

            # Forward to get output
            outputs, predict = model(inputs)
            # print(outputs)
            # Calculate Loss
            # loss = lossMSE(labels, outputs)
            # loss = lossNLLL(predict, labels)
            loss = lossCEL(outputs, target)
            loss.backward()

            # Updating parameters
            optimizer.step()

            # 输出统计
            running_loss += loss.item()

            if idx % 2000 == 1999:  # 每2000 mini-batchs输出一次
                print('[%d, %5d] loss: %.4f' % (epoch + 1, idx + 1, running_loss / 2000))

                running_loss = 0.0
    if opt.output_file is None:  # 输出文件路径未指定
        parameter = model.save()
        print("模型参数的路径:%s" % parameter)
    # print('epoch {}, loss {}'.format(epoch, loss.item()))
