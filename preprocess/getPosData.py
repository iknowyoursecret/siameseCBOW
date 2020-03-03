def get_postive_data_ls(idx, n_pos):
    """
    根据正例个数，获取正例集
    :param idx:主句的索引
    :param n_pos:正例个数
    :return:正例集（里边是句子的索引）
    """
    div, mod = divmod(n_pos, 2)
    if mod:  # 当正例集是单数时
        # div, mod = divmod(n_pos, 2)
        n_pos_ls = [pos_idx for pos_idx in range(idx - div, idx + div + 2)]
    else:
        n_pos_ls = [pos_idx for pos_idx in range(idx - div, idx + div + 1)]
    n_pos_ls.remove(idx)
    return n_pos_ls
