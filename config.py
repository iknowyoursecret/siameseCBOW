# coding:utf8
import warnings



class DefaultConfig(object):
    data_path = r'data/诡秘之主.txt'
    # 是否有训练好的参数
    # model_param_path = None
    model_param_path = r'output/SiasemeCBOW0219_14_46_43.pth'
    # 字典
    word_dic = None

    # 由字典的长度决定
    input_dim = None
    # 词嵌入的输输出维度
    output_dim = 100
    # 句子对齐的长度
    sent_length = 32
    # 正例集个数
    n_pos = 2
    # 反例集个数
    n_neg = 5

    # 学习参数设定
    learning_rate = 0.0001  # initial learning rate

    epochs = 5

    # max_epoch = 10
    # lr = 0.001  # initial learning rate
    # lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    # weight_decay = 0e-5  # 损失函数

    # 保存文件名
    output_file = None

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        # opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


opt = DefaultConfig()
