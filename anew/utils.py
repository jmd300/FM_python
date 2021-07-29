# -*- coding: utf-8 -*-

from math import exp
import numpy as np


def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")


def print_sample_rate(_data_set_name, _data_set_label):
    train_positive_num = 0
    validate_positive_num = 0

    for label in _data_set_label:
        if label == 1:
            train_positive_num += 1
    for label in _data_set_label:
        if label == 1:
            validate_positive_num += 1
    rate = round(train_positive_num / len(_data_set_label), 4)

    print("\n%s size is %d" % (_data_set_name, len(_data_set_label)))
    print("positive_num is %d, %f" % (train_positive_num, rate))
    print("negative_num is %d, %f" % (len(_data_set_label) - train_positive_num, 1 - rate))


def sigmoid(inx):
    # return 1.0 / (1 + exp(-inx))
    return 1. / (1. + exp(-max(min(inx, 15.), -15.)))


def load_data_set(_data_path, _have_column_name=True, _separator=',', _label_type=0):
    data_mat = []
    label_mat = []

    with open(_data_path) as data_file:
        all_data_lines = data_file.readlines()[1:] if _have_column_name else data_file.readlines()

        for line in all_data_lines:
            current_line = list(map(float, line.strip().split(_separator)))
            data_mat.append(current_line[:-1])
            if _label_type == 0:
                label_mat.append(float(current_line[-1]) * 2 - 1)
            else:
                label_mat.append(float(current_line[-1]))

    return np.mat(data_mat), label_mat


# 在传递的数据集引用的位置上使用train_data的各个特征最大最小值，直接进行归一化
def min_max_normalization_data_set(_data_set, _feature_value_max, _feature_value_min):
    data_num, feature_num = np.shape(_data_set)
    for i in range(data_num):
        for j in range(feature_num):
            x_max = _feature_value_max[j]
            x_min = _feature_value_min[j]
            if x_max == x_min:
                _data_set[i, j] = 1.0
            else:
                x = _data_set[i, j]
                _data_set[i, j] = round((x - x_min) / (x_max - x_min), 4)