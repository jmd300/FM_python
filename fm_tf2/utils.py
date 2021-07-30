from math import exp
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def sigmoid(inx):
    # return 1.0 / (1 + exp(-inx))
    return 1. / (1. + exp(-max(min(inx, 15.), -15.)))


def data_set_preprocessing(file_path, _dense_index, _sparse_index, test_size=0):
    data = pd.read_csv(file_path)
    row_num, col_num = data.shape

    data.columns = [str(i) for i in range(col_num)]  # 更改/添加Frame的列名

    data.dropna()  # 缺失值丢弃

    # 数值型特征归一化，如果测试集不使用训练集最大值最小值列表，准确率不好
    dense_features = [str(i) for i in _dense_index]
    data[dense_features] = MinMaxScaler().fit_transform(data[dense_features])

    # 类别型特征进行one_hot编码
    # sparse_features = [str(i) for i in _sparse_index]
    data = pd.get_dummies(data)

    label_index = str(data.columns[-1])
    X = data.drop([label_index], axis=1).values
    y = data[label_index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return (X_train, y_train), (X_test, y_test)
