from fm_tf2.utils import data_set_preprocessing

file_path = '../data/cancer_train.csv'

dense_index = [i for i in range(0, 30)]
sparse_index = []


(X_train, y_train), (X_test, y_test) = data_set_preprocessing(file_path, dense_index, sparse_index, test_size=0.5)
