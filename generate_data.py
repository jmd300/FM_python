from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

cancer_data = load_breast_cancer()["data"]
cancer_label = load_breast_cancer()["target"]

# 分为训练集与测试集
train_mat, test_mat = train_test_split(cancer_data, test_size=0.2, random_state=42)
train_label, test_label = train_test_split(cancer_label, test_size=0.2, random_state=42)

train_label = train_label.reshape(-1, 1)
test_label = test_label.reshape(-1, 1)

train_data = np.concatenate((train_mat, train_label), axis=1)
test_data = np.concatenate((test_mat, test_label), axis=1)

train_data = np.round(train_data, decimals=4)
test_data = np.round(test_data, decimals=4)

np.savetxt(r'/data/cancer_train.csv', train_data, delimiter=',', fmt='%.04f')
np.savetxt(r'/data/cancer_test.csv', test_data, delimiter=',', fmt='%.04f')




