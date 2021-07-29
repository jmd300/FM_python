# -*- coding: utf-8 -*-

from tqdm import tqdm
from anew.FM import FM
from anew.utils import *


if __name__ == '__main__':
    train_data = "data/cancer_train.csv"
    validate_data = "data/cancer_validate.csv"

    fm = FM()
    fm.load_train_data_set(train_data)
    fm.load_validate_data_set(validate_data)
    fm.min_max_normalization()

    # 看一下test_label中正负样本比例
    fm.print_data_set_info()

    # 是否进行网格式搜索最优超参数
    grid_test = False
    if grid_test:
        best_accuracy = 0
        best_epoch = 0
        best_learning_rate = 0
        best_dimension_num = 0
        grid_record = []
        print("fm.feature_num is ", fm.feature_num)
        for epoch in tqdm([i for i in range(1, 10)] + [i for i in range(10, 200, 5)]):
            for learning_rate in np.arange(0.001, 0.01, 0.002):
                for dimension_num in range(2, int(2 * fm.feature_num / 3), 2):
                    fm.fit(dimension_num, epoch, learning_rate)

                    accuracy_train = fm.get_accuracy(False)
                    accuracy_test = fm.get_accuracy()

                    print("epoch, dimension_num, learning_rate is ", epoch, dimension_num, learning_rate)
                    print("accuracy_train is ", accuracy_train)
                    print("accuracy_test is ", accuracy_test)

                    grid_record.append((epoch, learning_rate, dimension_num, accuracy_test))
                    if best_accuracy < accuracy_test:
                        best_accuracy = accuracy_test
                        best_epoch = epoch
                        best_learning_rate = learning_rate
                        best_dimension_num = dimension_num
        text_save("../data/grid_record.txt", grid_record)
        print("best is ", best_accuracy, best_epoch, best_learning_rate)
    else:
        fm.fit(_epoch=2, _learning_rate=0.01, _dimension_num=30)
        accuracy_train = fm.get_accuracy(False)
        accuracy_test = fm.get_accuracy()
        print("accuracy_train is ", accuracy_train)
        print("accuracy_test is ", accuracy_test)
