from fm_tf2.model import FM
from fm_tf2.utils import *
from itertools import product
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
from sklearn.metrics import accuracy_score
import argparse
import os
from numpy import arange


parser = argparse.ArgumentParser(description='命令行参数')
parser.add_argument('-k', type=int, help='v_dim', default=6)
parser.add_argument('-w_reg', type=float, help='w正则', default=1e-4)
parser.add_argument('-v_reg', type=float, help='v正则', default=1e-4)
args = parser.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == '__main__':
    cancer_all_path = '../data/cancer_all.csv'

    dense_index = [i for i in range(0, 30)]
    sparse_index = []
    (X_train, y_train), (X_test, y_test) = data_set_preprocessing(cancer_all_path, dense_index, sparse_index, 0.3)

    print("X_train is\n", X_train)
    print("y_train is\n", y_train)
    print("X_test is\n", X_test)
    print("y_test is\n", y_test)

    count = 0
    for x in y_train:
        if x == 1:
            count += 1
    print("In y_train label == 1's num is ", count)
    print("In y_train label == 0's num is ", len(y_train) - count)

    count = 0
    for x in y_test:
        if x == 1:
            count += 1
    print("In y_test label == 1's num is ", count)
    print("In y_test label == 0's num is ", len(y_test) - count)

    # k = args.k
    w_reg = args.w_reg
    v_reg = args.v_reg

    best_k = 0
    best_l = 0
    best_e = 0
    best_score = 0

    k_list = [8]
    l_list = [0.008]
    epochs_list = [8]
    '''k_list = [i for i in range(7, 11, 1)]
    l_list = [i for i in arange(0.007, 0.011, 0.0001)]
    epochs_list = [i for i in range(7, 11, 1)]'''
    for k, learning_rate, epochs in product(k_list, l_list, epochs_list):
        print("k, learning_rate, epochs is ", k, learning_rate, epochs)
        model = FM(k, w_reg, v_reg)
        optimizer = optimizers.SGD(learning_rate)

        summary_writer = tf.summary.create_file_writer('tensorboard')
        for i in range(epochs):
            with tf.GradientTape() as tape:
                y_pre = model(X_train)
                loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train, y_pred=y_pre))
                # print(loss.numpy())
            '''with summary_writer.as_default():
                tf.summary.scalar("loss", loss, step=i)'''
            grad = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))

        pre = model(X_test)

        pre = [1 if x > 0.5 else 0 for x in pre]
        accuracy = accuracy_score(y_test, pre)
        print("accuracy_score: ", accuracy)
        if accuracy > best_score:
            best_k = k
            best_l = learning_rate
            best_e = epochs
            best_score = accuracy
    print("best_k, best_l, best_e, best_score is ", best_k, best_l, best_e, best_score)
