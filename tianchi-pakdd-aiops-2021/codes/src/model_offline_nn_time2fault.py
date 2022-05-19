#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202104162357
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

"""
本模块（model_offline_nn_time2fault.py）利用召回模型召回的样本，训练Neural Network故障时间分类模型。
"""

import gc
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from numba import njit
from sklearn.metrics import (auc, mean_absolute_error, mean_squared_error,
                             precision_recall_curve, r2_score, roc_auc_score)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import (Add, BatchNormalization, Dense, Dropout,
                                     Input, PReLU)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from utils import (LiteModel, LoadSave, compute_custom_time2fault_metric,
                   make_df_evaluation, njit_f1, njit_infer_time2fault, sigmoid)

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 2
np.random.seed(GLOBAL_RANDOM_SEED)
tf.random.set_seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings("ignore")

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
###############################################################################
def njit_search_best_threshold_f1(
    y_true, y_pred_proba, low, high, n_search=256):
    """通过阈值枚举搜索最优的F1的阈值, 采用@njit技术加速计算"""
    # 依据y_pred_proba,确定可能的搜索范围
    unique_proba = np.unique(y_pred_proba)
    if len(unique_proba) < n_search:
        threshold_array = np.sort(unique_proba)
    else:
        threshold_array = np.linspace(low, high, n_search)

    # 遍历threshold_array, 进行阈值搜索
    best_f1, best_threshold = 0, 0

    f1_scores_list = []
    precision_scores_list = []
    recall_scores_list = []

    for threshold in threshold_array:
        f1_tmp, precision_tmp, recall_tmp = njit_f1(
            y_true, y_pred_proba, threshold)

        if f1_tmp > best_f1:
            best_f1 = f1_tmp
            best_threshold = threshold

        f1_scores_list.append(f1_tmp)
        precision_scores_list.append(precision_tmp)
        recall_scores_list.append(recall_tmp)

    return best_f1, best_threshold


def build_model(verbose=False, is_compile=True, **kwargs):
    """构建与编译用于Multi-label问题的全连接Neural Network。

    @Parameters:
    ----------
    verbose: {bool-like}
        决定是否打印模型结构信息。
    is_complie: {bool-like}
        是否返回被编译过的模型。
    **kwargs:
        其他重要参数。

    @Return:
    ----------
    Keras模型实例。
    """

    # 网络关键参数与输入
    # -------------------
    n_feats = kwargs.pop("n_feats", 128)
    n_classes = kwargs.pop("n_classes", 20)
    layer_input = Input(
        shape=(n_feats, ),
        dtype="float32",
        name="layer_input")

    # bottleneck网络结构
    # -------------------
    layer_dense_init = Dense(16, activation="relu")(layer_input)

    layer_dense_norm = BatchNormalization()(layer_dense_init)
    layer_dense_dropout = Dropout(0.1)(layer_dense_norm)
    layer_dense = Dense(8, activation="relu")(layer_dense_dropout)
    layer_dense_prelu = PReLU()(layer_dense)

    layer_dense_norm = BatchNormalization()(layer_dense)
    layer_dense_dropout = Dropout(0.1)(layer_dense_norm)
    layer_dense = Dense(16, activation="relu")(layer_dense_dropout)
    layer_dense_prelu = PReLU()(layer_dense)

    # 局部残差连接
    # -------------------
    layer_total = Add()([layer_dense_init, layer_dense_prelu])
    layer_pred = Dense(
        n_classes, activation="sigmoid", name="layer_output")(layer_total)
    model = Model([layer_input], layer_pred)

    if verbose:
        model.summary()
    if is_compile:
        model.compile(
            loss="binary_crossentropy", optimizer=Adam(0.0025),
            metrics=[tf.keras.metrics.AUC(
                num_thresholds=200, curve="ROC", name="auc")])
    return model


if __name__ == "__main__":
    # 全局化参数与待保存结果
    # ===============================
    # 利用多路召回模型，召回潜在故障样本，基于给定阈值判断样本是否需要被召回
    # 低阈值，强召回，可以获得更高的分数
    RECALL_THRESHOLD = 0.0183195

    # 只挑选小于THRESHOLD的样本进行时间预测模型的训练
    N_TIME2FAULT_MODELS = 1
    MAX_DAY2FAULT = 0.25

    n_classes = 10
    min_fault_minutes, fault_time_gap = 15, 25

    TRAIN_MONTH = [1, 2, 3, 4]
    VALID_MONTH = [5, 7]

    # 全局化的故障时间预测模型
    TIME2FAULT_PRED_MODELS = []
    TIME2FAULT_SCALER = []

    # 载入所有数据集，并按时间顺序进行排序
    # ===============================
    file_processor = LoadSave(dir_name="./data_tmp/", verbose=1)
    total_feats_df = file_processor.load_data(
        file_name="xgb_total_feats_df.pkl")
    failure_tag = file_processor.load_data(
        dir_name="./data_tmp/", file_name="failure_tag.pkl")

    # 全特征数据预处理
    total_feats_df.sort_values(
        by=["collect_time_bin_right_edge", "serial_number"],
        inplace=True, ascending=[True, True])
    total_feats_df.reset_index(inplace=True, drop=True)

    # 故障表数据预处理
    failure_tag["min_range"] = (failure_tag["failure_time"] + \
                                pd.Timedelta(-7, unit="d")).astype(int) / 10**9
    failure_tag["max_range"] = failure_tag["failure_time_unix"].values

    # 检测是否有GPU
    gpus = tf.config.experimental.list_physical_devices("GPU")

    # 全局主键
    KEY_COLS = [
        "serial_number", "month", "global_day",
        "collect_time_bin_right_edge",
        "days_to_failure", "hours_to_failure"]

    # Recall stage：召回所有可能故障的样本集
    # （即大于召回模型预测阈值则保留；否则舍去。等价于基于阈值的初筛，此处只载入数据。）
    # ===============================
    file_processor = LoadSave(dir_name="./models/")
    recall_meta_df, recall_threshold_df = file_processor.load_data(
        file_name="recall_xgb_clf_meta.pkl")
    recall_bool = recall_meta_df["recall_proba"] >= RECALL_THRESHOLD

    file_processor = LoadSave(dir_name="./models/")
    recall_meta, recall_threshold_df = file_processor.load_data(
        file_name="recall_xgb_clf_meta.pkl")

    # 选择train和test被召回出来的样本
    recalled_samples_bool = np.ones(len(total_feats_df))
    curr_recalled_samples_bool = \
        recall_meta["recall_proba"] >= RECALL_THRESHOLD

    recalled_samples_bool = np.logical_and(
        recalled_samples_bool, curr_recalled_samples_bool)

    # 只保留被召回的样本进行训练，可以选择不将1月份数据加入训练集
    total_feats_df = \
        total_feats_df[recalled_samples_bool].reset_index(drop=True)

    del recall_meta, recalled_samples_bool
    gc.collect()

    # Time2fault prediction stage：故障时间预测模型
    # ===============================
    train_bool = \
        (total_feats_df["days_to_failure"] <= MAX_DAY2FAULT) & \
        (total_feats_df["days_to_failure"].notnull())

    # 训练数据meta information准备
    # -------------------
    train_meta_df = total_feats_df[train_bool][KEY_COLS].reset_index(drop=True)
    train_feats = total_feats_df[train_bool].drop(KEY_COLS, axis=1).values

    train_group_ids = pd.to_datetime(
        train_meta_df["collect_time_bin_right_edge"], unit="s")

    del total_feats_df
    gc.collect()

    # 制作分类器标签
    # -------------------
    minutes2fault_threshold_array = [
        min_fault_minutes + i * fault_time_gap for i in range(n_classes)]

    for idx, minute_threshold in enumerate(minutes2fault_threshold_array):
        train_meta_df["class_{}".format(idx)] = \
            ((train_meta_df["hours_to_failure"] * 60) \
                <= minute_threshold).astype(int)

    # 按Time Series Split进行模型训练
    # -------------------

    train_bool = np.logical_or.reduce(
        [train_meta_df["month"].values == month for month in TRAIN_MONTH])
    valid_bool = np.logical_or.reduce(
        [train_meta_df["month"].values == month for month in VALID_MONTH])

    train_idx = np.arange(0, len(train_meta_df))[train_bool]
    valid_idx = np.arange(0, len(train_meta_df))[valid_bool]

    # STEP 1: 依据训练与验证的bool索引, 构建训练与验证对应的数据集
    # -------------------
    X_train = train_feats[train_idx]
    X_valid = train_feats[valid_idx]

    target_names = [item for item in train_meta_df.columns \
        if "class" in item]
    y_train = train_meta_df.iloc[train_idx][target_names].values
    y_valid = train_meta_df.iloc[valid_idx][target_names].values

    # STEP 2: 对训练数据进行归一化，并应用于validation数据上
    # -------------------
    X_sc = StandardScaler()
    X_sc.fit(X_train)

    X_train_scaled = X_sc.transform(X_train)
    X_valid_scaled = X_sc.transform(X_valid)

    # STEP 3: 构建与编译神经网络模型
    # -------------------
    print("[INFO] {} NN Time2fault Model Training Start:".format(
        str(datetime.now())[:-4]))
    print("[INFO] train shape: {}, valid shape: {}, "
          "time2fault models: {}".format(
              X_train.shape, X_valid.shape, N_TIME2FAULT_MODELS))
    print("=========================================")

    # 清理显存中的计算图
    K.clear_session()
    gc.collect()

    nn_clf = build_model(
        n_classes=n_classes, n_feats=X_train.shape[1])

    early_stop = EarlyStopping(
        monitor="val_auc", mode="max",
        verbose=1, patience=50,
        restore_best_weights=True)

    history = nn_clf.fit(
        x=[X_train_scaled], y=y_train,
        batch_size=1024,
        epochs=450,
        validation_data=([X_valid_scaled], y_valid),
        callbacks=[early_stop],
        verbose=0)

    # STEP 4: 计算validation数据的time2fault时间，并搜索每个类别最优切分阈值
    # -------------------
    valid_pred_proba = nn_clf.predict(X_valid_scaled)

    valid_threhsold_list = []
    valid_best_f1_list = []
    for i in range(n_classes):
        best_f1, best_threshold = njit_search_best_threshold_f1(
            y_valid[:, i], valid_pred_proba[:, i],
            low=0, high=0.9, n_search=512)
        valid_threhsold_list.append(best_threshold)
        valid_best_f1_list.append(best_f1)

    # 保存归一化工具与决策阈值列表
    TIME2FAULT_SCALER = [X_sc, valid_threhsold_list]

    # 打印Cross validation的结果
    roc_auc_vals = roc_auc_score(y_valid, valid_pred_proba, average="macro")
    print("-- fold {}({}) valid pr auc: {:.4f}".format(
        0+1, N_TIME2FAULT_MODELS, roc_auc_vals))
    print("-- fold {}({}) avg f1: {:.5f}, min f1: {:.5f}, "
          "max f1: {:.5f}".format(
              0+1, N_TIME2FAULT_MODELS,
              np.mean(valid_best_f1_list),
              np.min(valid_best_f1_list),
              np.max(valid_best_f1_list)))
    print("=========================================")

    # 保存召回模型和对应的归一化工具
    # -------------------
    file_processor = LoadSave(dir_name="./models/")
    file_processor.save_data(
        file_name="time2fault_nn_clf_scalers.pkl",
        data_file=TIME2FAULT_SCALER)

    nn_clf.save("./models/time2fault_nn_clf_models")

    # 搜索最优的时间预测结果
    valid_pred_label = []
    for i in range(n_classes):
        valid_pred_label.append(
        (valid_pred_proba[:, i] >= valid_threhsold_list[i]).reshape(-1, 1))
    valid_pred_label = np.hstack(valid_pred_label).astype(int)

    # # 搜索最优的时间预测结果
    # valid_pred_label = (valid_pred_proba > 0.5).astype(int)

    time2fault_minutes_array = njit_infer_time2fault(
        valid_pred_label, min_fault_minutes=1, gap_minutes=15)
    total_res = np.hstack(
        [time2fault_minutes_array, (train_meta_df["hours_to_failure"].iloc[valid_idx].values * 60).reshape(-1, 1)])

    pred_accuracy = 1 - np.sum(total_res[:, 0] > total_res[:, 1]) / len(total_res)
    pred_score = compute_custom_time2fault_metric(total_res[:, 0], total_res[:, 1])

    print("-- fold {}({}): accuracy: {:.6f}, avg custom score: {:.6f}".format(
        0, N_TIME2FAULT_MODELS,
            pred_accuracy, pred_score))
