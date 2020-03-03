def sentence2idx(dat_str, word_dic):
    """

    :param dat_str:输入的每句话
    :param word_dic:字典
    :return:将每句话对应成数字
    """

    sentence_num_sequences = [word_dic[word] for word in dat_str]
    return sentence_num_sequences


def padding(line, seq_length, unk):
    """
    句子对齐
    :param line: 句子
    :param seq_length: 用于对其的长度
    :param unk: 填充值
    :return:
    """
    if len(line) < seq_length:
        line.extend([unk] * (seq_length - len(line)))
    else:
        line = line[:seq_length]
    return line


def data2num_list(dat, word_dic, sent_length):
    """
    全部数据格式转换成longtensor
    :param dat:
    :return:
    """
    return [padding(sentence2idx(line, word_dic), sent_length, word_dic['unk']) for line in dat]


def sig_data2num_list(single_dat, word_dic, sent_length):
    """
    单个数据格式转换成longtensor
    :param single_dat:单个数据
    :return:longtensor类型的tensor
    """
    return padding(sentence2idx(single_dat, word_dic), sent_length, word_dic['unk'])
