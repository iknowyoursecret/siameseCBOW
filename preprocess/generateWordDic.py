def generate_word_dic(corpus):
    """
    根据语料生成字的词频字典及字对应字典
    :param corpus:语料
    :return:字典
    """
    # 构建字典
    word_fre = {}  # 真~字典 value值为词频
    for sentence in corpus:
        for word in sentence:
            if word not in word_fre.keys():
                word_fre[word] = 0
            else:
                word_fre[word] += 1

    word_dic = {word: idx for idx, word in enumerate(word_fre.keys())}  # value值为对应int值
    num_unk = len(word_dic)
    word_dic['unk'] = num_unk
    return word_fre, word_dic
