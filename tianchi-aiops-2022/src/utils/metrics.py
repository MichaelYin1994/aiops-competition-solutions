
# -*- coding: utf-8 -*-

# Created on 202203080113
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
对于官方Metrics的实现。
'''

import numpy as np
import tensorflow as tf
from numba import njit


def compute_custom_score(y_true_label, y_pred_label):
    '''按照官方要求计算Weighted F1 Score'''
    weighted_mean_score = 0
    weights = [5 / 11, 4 / 11, 1 / 11, 1 / 11]

    for i, label in enumerate([0, 1, 2, 3]):
        f1_score, _, _ = njit_f1(
            (y_true_label == label).astype(int),
            (y_pred_label == label).astype(int)
        )
        weighted_mean_score += f1_score * weights[i]

    return weighted_mean_score


@njit
def njit_f1(y_true_label, y_pred_label):
    '''计算F1分数，使用njit加速计算'''
    # https://www.itread01.com/content/1544007604.html
    tp = np.sum(
        np.logical_and(
            np.equal(y_true_label, 1),
            np.equal(y_pred_label, 1)
        )
    )
    fp = np.sum(
        np.logical_and(
            np.equal(y_true_label, 0),
            np.equal(y_pred_label, 1)
        )
    )
    # tn = np.sum(np.logical_and(np.equal(y_true, 1),
    #                            np.equal(y_pred_label, 0)))
    fn = np.sum(
        np.logical_and(
            np.equal(y_true_label, 1),
            np.equal(y_pred_label, 0)
        )
    )

    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall


class CustomMetrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data=None, **kwargs):
        super(CustomMetrics, self).__init__()
        self.valid_data = valid_data

    def on_epoch_end(self, epoch, logs):
        '''预测每个batch的标签，并计算最后的F1分数'''

        # 获取预测的结果
        y_true_label_list, y_pred_label_list = [], []
        for data_pack in self.valid_data:
            X_batch, y_true_batch = data_pack[0], data_pack[1]
            y_pred_proba_batch = self.model.predict(X_batch)

            y_true_label_list.append(
                np.argmax(y_true_batch, axis=1)
            )
            y_pred_label_list.append(
                np.argmax(y_pred_proba_batch, axis=1)
            )

        y_true_label_array = np.concatenate(y_true_label_list)
        y_pred_label_array = np.concatenate(y_pred_label_list)

        score = compute_custom_score(y_true_label_array, y_pred_label_array)
        logs['val_custom'] = score

        return score
