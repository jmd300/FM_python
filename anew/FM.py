# -*- coding: utf-8 -*-

from __future__ import division
from random import normalvariate  # 正态分布
from sklearn import preprocessing
from tqdm import tqdm
from anew.utils import *

'''
    data : 数据
    dimension_num : 潜在分解维度数
    learning_rate ： 学习速率
    epoch ： 迭代次数
    _w,_w_0,_v ： 拆分子矩阵的weight
    have_column_name : 是否带有columns_name
'''


class FM(object):
    def __init__(self):
        self.epoch = None
        self.learning_rate = None
        self.dimension_num = None

        self.train_data_path = None
        self.validate_data_path = None

        self.train_data = None
        self.train_label = None

        self.validate_data = None
        self.validate_label = None

        self.feature_num = 0

        self.w = None
        self.w_0 = None
        self.v = None

        self.train_feature_value_max = None
        self.train_feature_value_min = None

    # follow_train: test数据的归一化是否以train数据的最大最小值进行归一化
    def min_max_normalization(self, follow_train=True):
        min_max_normalization_data_set(self.train_data, self.train_feature_value_max, self.train_feature_value_min)
        if follow_train:
            min_max_normalization_data_set(self.validate_data, self.train_feature_value_max, self.train_feature_value_min)
        else:
            min_max_scaler = preprocessing.MinMaxScaler()
            self.validate_data = min_max_scaler.fit_transform(self.validate_data)

    # 使用前先将数据集文件转换为feature1, feature2, label, ...的格式，label为[0， 1]
    def load_train_data_set(self, _data_path):
        self.train_data, self.train_label = load_data_set(_data_path, False)
        _, self.feature_num = np.shape(self.train_data)
        self.train_feature_value_max = np.max(np.asarray(self.train_data), axis=0)
        self.train_feature_value_min = np.min(np.asarray(self.train_data), axis=0)

    def load_validate_data_set(self, _data_path):
        self.validate_data, self.validate_label = load_data_set(_data_path, False)

    # 得到对应的特征weight的矩阵
    def fit(self, _dimension_num, _epoch, _learning_rate=0.01, _incremental=False):
        self.epoch = _epoch
        self.learning_rate = _learning_rate
        self.dimension_num = _dimension_num

        data_nums, feature_num = np.shape(self.train_data)

        # 感觉这里的初始化方式也可以变化的，当然现在效果挺好的
        w_0 = 0.0
        w = np.zeros((feature_num, 1))
        v = normalvariate(0, 0.2) * np.ones((feature_num, self.dimension_num))

        for _ in tqdm(range(self.epoch)):
            # 每次使用一个样本优化
            for x in range(data_nums):
                # 计算逻辑参考：http://blog.csdn.net/google19890102/article/details/45532745
                # xi·vi,xi与vi的矩阵点积
                inter_1 = self.train_data[x] * v

                # xi与xi的对应位置乘积   与   xi^2与vi^2对应位置的乘积    的点积
                inter_2 = np.multiply(self.train_data[x], self.train_data[x]) * np.multiply(v, v)

                # 完成交叉项,xi*vi*xi*vi - xi^2*vi^2
                interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.0
                # 计算预测的输出
                p = w_0 + self.train_data[x] * w + interaction

                # 计算sigmoid(y*pred_y)-1，label是[-1, 1]
                loss = sigmoid(self.train_label[x] * p[0, 0]) - 1

                # 更新参数
                w_0 = w_0 - self.learning_rate * loss * self.train_label[x]
                for i in range(feature_num):
                    if self.train_data[x, i] != 0:
                        w[i, 0] = w[i, 0] - self.learning_rate * loss * self.train_label[x] * self.train_data[x, i]
                        for j in range(self.dimension_num):
                            v[i, j] = v[i, j] - self.learning_rate * loss * self.train_label[x] * \
                                      (self.train_data[x, i] * inter_1[0, j] -
                                       v[i, j] * self.train_data[x, i] * self.train_data[x, i])
        self.w_0, self.w, self.v = w_0, w, v

    def get_accuracy(self, is_validate_data=True):
        _validate_data = self.validate_data if is_validate_data else self.train_data
        _validate_label = self.validate_label if is_validate_data else self.train_label

        w_0, w, v = self.w_0, self.w, self.v
        data_nums, feature_num = np.shape(_validate_data)

        item_count = 0
        error = 0

        for x in range(data_nums):
            item_count += 1
            inter_1 = _validate_data[x] * v
            inter_2 = np.multiply(_validate_data[x], _validate_data[x]) * np.multiply(v, v)
            interaction = sum(np.multiply(inter_1, inter_1) - inter_2) / 2.0
            p = w_0 + _validate_data[x] * w + interaction
            pre = sigmoid(p[0, 0])
            if pre < 0.5 and _validate_label[x] == 1.0:
                error += 1
            elif pre >= 0.5 and _validate_label[x] == -1.0:
                error += 1
            else:
                continue
        value = 1 - float(error) / item_count
        return value

    def print_data_set_info(self):
        print_sample_rate("train_data", self.train_label)
        print_sample_rate("validate_data", self.validate_label)


class NotFittedError(Exception):
    """
    Exception class to raise if estimator is used before fitting
    """
    pass
