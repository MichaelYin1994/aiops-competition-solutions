#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202103201115
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

"""
本模块（model_offline_xgb_recall.py）采用多路召回策略，利用历史月份的
日志特征数据作为train set，其他月份作为valid set，依据参数与随机种子
不同，训练多组XGBoost的召回模型；并将召回模型用于召回可能异常的样本。
"""

import gc
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from tqdm import tqdm

from utils import LoadSave, njit_f1

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 1
np.random.seed(GLOBAL_RANDOM_SEED)
tf.random.set_seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings("ignore")
###############################################################################

def custom_precision_recall_auc(y_pred_proba, dtrain):
    """针对分类问题的precision与recall的auc优化Metric。"""
    y_true_label = dtrain.get_label()

    precision_array, recall_array, _ = \
        precision_recall_curve(
            y_true_label.reshape(-1, 1),
            y_pred_proba.reshape(-1, 1))

    auc_score = auc(recall_array, precision_array)
    return "pr_auc", -1 * auc_score


if __name__ == "__main__":
    # 全局化参数与待保存结果
    # ===============================
    # 全局召回模型的路数
    # -------------------
    IS_DEBUG = False
    N_RECALL_MODELS = 3

    # 召回模型训练数据集与测试集范围
    # -------------------
    TRAIN_MONTH = [1, 2, 3, 4]
    VALID_MONTH = [5, 7]

    # 召回距离故障天数小于该阈值的样本
    MAX_DAY2FAULT = 0.25

    # 待保存召回模型
    # -------------------
    TRAINED_RECALL_MODELS = []

    # 载入所有数据集，并按时间顺序进行排序
    # ===============================
    file_processor = LoadSave(dir_name="./data_tmp/", verbose=1)
    total_feats_df = file_processor.load_data(
        file_name="xgb_total_feats_df.pkl")

    total_feats_df.sort_values(
        by=["collect_time_bin_right_edge", "serial_number"],
        inplace=True, ascending=[True, True])
    total_feats_df.reset_index(inplace=True, drop=True)

    # 检测是否有GPU
    gpus = tf.config.experimental.list_physical_devices("GPU")

    # 训练数据,测试数据与targets准备
    # -------------------
    KEY_COLS = [
        "serial_number", "month", "global_day",
        "collect_time_bin_right_edge",
        "days_to_failure", "hours_to_failure"]
    total_meta_df = total_feats_df[KEY_COLS].reset_index(drop=True)

    # 划分训练与测试数据特征
    # -------------------
    total_feats = total_feats_df.drop(KEY_COLS, axis=1).values

    del total_feats_df
    gc.collect()

    # Recall Stage：多路召回模型，优化整体分类的AUC
    # - 召回时，可只选择部分特征进行召回模型训练，强化模型鲁棒性
    # - 提供指标，监控召回模型的效果
    # ===============================

    # 准备用于召回模型训练的数据
    # -------------------
    recall_train_bool = np.logical_or.reduce(
        [total_meta_df["month"].values == month for month in TRAIN_MONTH])
    recall_valid_bool = np.logical_or.reduce(
        [total_meta_df["month"].values == month for month in VALID_MONTH])

    recall_train_idx = \
        np.arange(0, len(total_feats))[recall_train_bool]
    recall_valid_idx = \
        np.arange(0, len(total_feats))[recall_valid_bool]

    recall_train_feats = total_feats[recall_train_idx]
    recall_valid_feats = total_feats[recall_valid_idx]

    recall_train_targets = \
        total_meta_df["days_to_failure"].iloc[recall_train_idx].values
    recall_valid_targets = \
        total_meta_df["days_to_failure"].iloc[recall_valid_idx].values

    recall_train_targets[np.isnan(recall_train_targets)] = MAX_DAY2FAULT
    recall_valid_targets[np.isnan(recall_valid_targets)] = MAX_DAY2FAULT
    recall_train_targets[recall_train_targets >= MAX_DAY2FAULT] = MAX_DAY2FAULT
    recall_valid_targets[recall_valid_targets >= MAX_DAY2FAULT] = MAX_DAY2FAULT

    recall_train_label = np.where(
        recall_train_targets < MAX_DAY2FAULT, 1, 0)
    recall_valid_label = np.where(
        recall_valid_targets < MAX_DAY2FAULT, 1, 0)

    recall_valid_sn = \
        total_meta_df.iloc[recall_valid_idx]["serial_number"].values
    recall_valid_time_stamp = \
        total_meta_df.iloc[recall_valid_idx]["collect_time_bin_right_edge"].values

    # 预备训练召回模型
    # -------------------
    print("[INFO] {} XGBoost Recall Model Training Start:".format(
        str(datetime.now())[:-4]))
    print("[INFO] train shape: {}, valid shape: {}, recall models: {}".format(
        recall_train_feats.shape, recall_valid_feats.shape, N_RECALL_MODELS))
    print("=========================================")

    recall_model_auc_list = []
    valid_pred_proba_list, total_pred_proba_list = [], []
    for i in range(N_RECALL_MODELS):

        # 随机化召回模型参数
        xgb_params = {
            "n_estimators": 5000,
            "max_depth": np.random.choice(
                [3, 4, 5]),
            "learning_rate": np.random.choice(
                [0.03, 0.04, 0.05, 0.06]),
            "verbosity": 0,
            "objective": "binary:logistic",
            "booster": "gbtree",
            "colsample_bytree": np.random.choice(
                [0.97, 0.98, 0.99]),
            "colsample_bylevel": np.random.choice(
                [0.97, 0.98, 0.99]),
            "subsample": np.random.choice(
                [0.97, 0.98, 0.985, 0.99]),
            "disable_default_eval_metric": 1,
            "random_state": np.random.randint(4096)}

        # 若有GPU资源则采用GPU计算
        if gpus:
            xgb_params["tree_method"] = "gpu_hist"
            xgb_params["gpu_id"] = 0

        # 训练召回模型
        print(xgb_params)
        xgb_recall = xgb.XGBClassifier(**xgb_params)
        xgb_recall.fit(
            recall_train_feats,
            recall_train_label,
            eval_set=[(recall_valid_feats,
                       recall_valid_label)],
            early_stopping_rounds=800,
            verbose=IS_DEBUG,
            eval_metric=custom_precision_recall_auc)

        valid_pred_proba = xgb_recall.predict_proba(
            recall_valid_feats,
            ntree_limit=xgb_recall.best_iteration+1)[:, 1]
        total_pred_proba = xgb_recall.predict_proba(
            total_feats,
            ntree_limit=xgb_recall.best_iteration+1)[:, 1]

        # 记录该模型指标，并打印指标
        precision_array, recall_array, _ = \
            precision_recall_curve(
                recall_valid_label.reshape(-1, 1),
                valid_pred_proba.reshape(-1, 1))
        recall_model_auc_list.append(
            auc(recall_array, precision_array))

        # 存储模型与特征ID
        valid_pred_proba_list.append(valid_pred_proba)
        total_pred_proba_list.append(total_pred_proba)

        TRAINED_RECALL_MODELS.append(xgb_recall)

        print("-- fold {}({}) valid pr auc: {:.4f}, best iters: {}".format(
            i+1, N_RECALL_MODELS, recall_model_auc_list[-1],
            xgb_recall.best_iteration+1))

    valid_pred_proba = np.mean(valid_pred_proba_list, axis=0)
    total_pred_proba = np.mean(total_pred_proba_list, axis=0)

    precision_array, recall_array, _ = \
        precision_recall_curve(
            recall_valid_label.reshape(-1, 1),
            valid_pred_proba.reshape(-1, 1))
    valid_total_eval_auc = auc(recall_array, precision_array)

    print("-- total mean precision-recall auc: {:.4f}, "
          "total eval precision-recall auc: {:.4f}".format(
              np.mean(recall_model_auc_list), valid_total_eval_auc))
    print("=========================================")

    # 选取分类切分阈值，保证2月的召回率，召回率考虑两方面因素：
    # - 尽量保证全样本上的F1率高（全样本的F1）
    # - 尽量保证不漏选机器（从Fault Machine的角度来进行工作）
    # -------------------
    score_total_list = []
    threshold_array = np.linspace(0, 0.25, 2048)
    total_meta_df["recall_proba"] = total_pred_proba
    total_sn = total_meta_df["serial_number"].values

    for month in [5, 7]:
        recall_idx = np.arange(0, len(total_feats))[
            total_meta_df["month"] == month]

        # 构建召回样本的标签与故障机器的sn
        recall_targets = \
            total_meta_df["days_to_failure"].iloc[recall_idx].values
        recall_targets[np.isnan(recall_targets)] = MAX_DAY2FAULT
        recall_label = (recall_targets < MAX_DAY2FAULT).astype(int)

        recall_sn = total_sn[recall_idx]
        recall_pred_proba = total_meta_df["recall_proba"].values[recall_idx]

        # 搜索最佳阈值
        score_tmp_list = []
        for threshold in tqdm(threshold_array):
            recall_pred_label = (recall_pred_proba > threshold).astype(int)

            # Consider 1: 全局的样本上的F1值要尽量高
            # -------------------
            recall_total_f1, _, _ = njit_f1(
                recall_label.reshape(-1, 1),
                recall_pred_proba.reshape(-1, 1),
                threshold)

            # Consider 2: 尽量让真正故障机器的召回率高
            # -------------------
            total_n_unique_machines = len(np.unique(recall_sn))
            true_fault_sn = set(np.unique(
                recall_sn[recall_label == 1]).tolist())
            pred_fault_sn = set(np.unique(
                recall_sn[recall_pred_label == 1]).tolist())
            recall_fault_machine = \
                len(true_fault_sn.intersection(pred_fault_sn)) \
                    / len(true_fault_sn)

            score_tmp_list.append(
                [recall_total_f1, recall_fault_machine,
                 len(pred_fault_sn) / total_n_unique_machines])
        score_total_list.append(np.array(score_tmp_list))

    score_total = pd.DataFrame(
        np.mean(score_total_list, axis=0),
        columns=["sample_f1_score", "machine_recall_score", "n_unique_pred_sn"])
    score_total["threshold"] = threshold_array

    # 保存召回模型和对应的DataFrame
    # -------------------
    file_processor = LoadSave(dir_name="./models/")
    file_processor.save_data(
        file_name="recall_xgb_clf_models.pkl",
        data_file=TRAINED_RECALL_MODELS)

    total_feats_recall_meta_df = [total_meta_df[[
        "serial_number", "collect_time_bin_right_edge",
        "recall_proba", "days_to_failure"]], score_total]
    file_processor.save_data(
        file_name="recall_xgb_clf_meta.pkl",
        data_file=total_feats_recall_meta_df)
