#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202104141624
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

"""
本模块（model_offline_xgb_reg.py）利用召回模型召回的样本，训练XGBoost回归模型。
训练采用Time Series Split的方式按月进行Split。此模块仅有train和valid步骤。
"""

import gc
import warnings
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)

from utils import (LiteModel, LoadSave, make_df_evaluation,
                   njit_infer_time2fault, sigmoid)

# 设定全局随机种子，并且屏蔽warnings
GPU_ID = 0
GLOBAL_RANDOM_SEED = 3
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

def yield_train_valid_range():
    """Generator of daily based cross validation"""
    cv_folds = [[[0, 150+1], [181, 211+1]],
                [[0, 120], [120, 150+1]],
                [[0, 90], [120, 150+1]],
                [[0, 211+1], [181, 211+1]],]

    # cv_folds = [[[0, 60], [120, 150+1]],
    #             [[0, 90], [120, 150+1]],
    #             [[0, 120], [120, 150+1]],
    #             [[0, 150+1], [181, 211+1]],
    #             [[0, 211+1], [181, 211+1]],]

    for cv_range in cv_folds:
        yield cv_range


def custom_score_metric(y_pred_time2fault, dtrain, sn_array, time_stamp_array,
                        custom_threshold, true_threshold, failure_tag_dict):
    """用于XGBoost早停策略的Metric。根据threshold，判断dtrain结果
    是否为正样本。随后依据官方Metric进行评分。

    @Parameters
    ----------
        y_pred_time2fault: {array-like}
            XGBoost的validation数据集预测结果，为官方Metric的PTI时间。
        dtrain: {xgboost data obj}
            用于获取实际标签。
        sn_array: {array-like}
            validation数据集的serial_number数组。
        pred_time_stamp_array: {array-like}
            作出预测的时刻的unix time stamp。
        threshold: {int-like}
            用于判断y_pred的标签的阈值。
        true_threshold: {int-like}
            用于判断y_true的标签的真实阈值。
    """
    y_pred_label = (y_pred_time2fault <= custom_threshold).astype(int)

    # STEP 1: 选择所有第1次被预测为Fault Mahcine的机器的index
    # -------------------
    # 只挑选被预测为故障的机器与其相关的一些特性
    y_pred_time2fault_tmp = y_pred_time2fault[y_pred_label == 1]
    sn_array_tmp = sn_array[y_pred_label == 1]
    time_stamp_array_tmp = time_stamp_array[y_pred_label == 1]

    # 按照时间顺序排序预测的结果
    sorted_time_stamp_idx = np.argsort(time_stamp_array_tmp)
    time_stamp_array_tmp = np.sort(time_stamp_array_tmp)

    y_pred_time2fault_tmp = y_pred_time2fault_tmp[sorted_time_stamp_idx]
    sn_array_tmp = sn_array_tmp[sorted_time_stamp_idx]

    # 挑选第1次被预测为故障的机器与其idx
    pred_fault_sn, pred_fault_sn_idx = np.unique(
        sn_array_tmp, return_index=True)

    pred_time2fault_array = y_pred_time2fault_tmp[pred_fault_sn_idx]
    pred_failure_date_array = time_stamp_array_tmp[pred_fault_sn_idx]

    # STEP 2: 计算validaiton上第二轮评估Metric的分数
    # -------------------
    # npr: 评估窗口内所有发生内存故障服务器的数量
    npr = len(failure_tag_dict)

    # 计算每一个预测的时间是否落在对应的failure time区间内
    y_pred_labels = np.zeros((len(pred_fault_sn))) - 1

    for i in range(len(pred_fault_sn)):
        # 获取当前的serial_number与当前的预测窗口left edge时间
        curr_sn = pred_fault_sn[i]
        curr_pred_date = pred_failure_date_array[i]

        # 若该sn位于failure_tag_dict中
        if curr_sn in failure_tag_dict and \
            curr_pred_date > failure_tag_dict[curr_sn][0] and \
                curr_pred_date < failure_tag_dict[curr_sn][1]:
            y_pred_labels[i] = 1
        else:
            y_pred_labels[i] = 0

    # ntpr: 评估窗口内发生内存故障的服务器被提前7天内发现的数量
    ntpr = np.sum(y_pred_labels)

    # 召回率
    if npr == 0:
        recall = 0
    else:
        recall = ntpr / npr

    # npp: 评估窗口内被预测出来未来7天会发生故障的服务器数量
    npp = len(y_pred_labels)

    # ntpp: 评估窗口内第一次预测故障时间后7天内确实发生故障的服务器数量(与ntpr一致)
    ntpp = ntpr

    # 精准率
    if npp == 0:
        precision = 0
    else:
        precision = ntpp / npp

    # f1分数
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return "custom_f1", -1 * f1


if __name__ == "__main__":
    # 全局化参数与待保存结果
    # ===============================
    # 利用多路召回模型，召回潜在故障样本，基于给定阈值判断样本是否需要被召回
    # 低阈值，强召回，可以获得更高的分数
    RECALL_THRESHOLD = 0.0183195

    # 只挑选小于THRESHOLD的样本进行时间预测模型的训练
    MAX_DAY2FAULT = 0.25

    # 通过参数计算预测的故障时间的范围
    MIN_FAULT_MINUTES, GAP_MINUTES = 1, 14
    MIN_THRESHOLD, MAX_THRESHOLD, N_SEARCH_THRESHOLD = 0.16, 0.195, 20

    # 故障样本预测模型的每一个FOLD的模型个数
    N_MODELS_PER_FOLD = 1

    SELECTED_INFER_PARAM_IDX = 11
    IS_TRAIN_FOR_INFER = True

    # 全局化的故障时间预测模型
    TIME2FAULT_JUDGE_MODELS = []

    # 载入预先训练好的time2fault时间预测模型
    file_processor = LoadSave(dir_name="./models/")
    time2fault_pred_model = \
        tf.keras.models.load_model("./models/time2fault_nn_clf_models")

    time2fault_scaler, time2fault_threshold_list = \
        file_processor.load_data(file_name="time2fault_nn_clf_scalers.pkl")

    # 载入所有数据集，并按时间顺序进行排序
    # ===============================
    file_processor = LoadSave(dir_name="./data_tmp/", verbose=0)
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

    del recall_meta
    gc.collect()

    # Time prediction stage：预测每个样本的故障时间
    # ===============================

    # 训练数据，测试数据与targets准备
    # -------------------
    train_month = [1, 2, 3, 4, 5, 7]
    train_bool = np.logical_or.reduce(
        [total_feats_df["month"].values == month for month in train_month])
    train_meta_df = total_feats_df[train_bool][KEY_COLS].reset_index(drop=True)

    # 训练数据meta information准备
    # -------------------
    train_feats = total_feats_df[train_bool]
    train_feats = train_feats.drop(KEY_COLS, axis=1).values

    del total_feats_df
    gc.collect()

    # Training stage：训练获得validation上官方Metric高的模型
    # ===============================
    train_group_ids = train_meta_df["global_day"].values

    # 开始基于固定阈值进行训练
    # -------------------
    print("[INFO] {} XGBoost Regressor training start:".format(
        str(datetime.now())[:-4]))
    print("[INFO] train shape: {}".format(train_feats.shape))
    print("=========================================")

    judgement_threshold_array = np.linspace(
        MIN_THRESHOLD, MAX_THRESHOLD, N_SEARCH_THRESHOLD)
    total_params_list = [
        {"n_estimators": 5000,
        "max_depth": np.random.choice(
            [4, 5, 6]),
        "learning_rate": np.random.choice(
            [0.04, 0.05, 0.06, 0.07]),
        "verbosity": 0,
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "colsample_bytree": np.random.choice(
            [0.98, 0.99]),
        "colsample_bylevel": np.random.choice(
            [0.98, 0.99]),
        "subsample": np.random.choice(
            [0.97, 0.98, 0.99]),
        "disable_default_eval_metric": 1,
        "random_state": np.random.randint(2048)} for _ in range(100)]

    # 确定参数list的列表
    params_list = []
    for i in range(len(judgement_threshold_array)):
        tmp_params = []
        for j in range(N_MODELS_PER_FOLD):
            tmp_params.append(total_params_list[
                int(i * N_MODELS_PER_FOLD) + j
            ])
        params_list.append(tmp_params)

    if IS_TRAIN_FOR_INFER:
        judgement_threshold_array = \
            [judgement_threshold_array[SELECTED_INFER_PARAM_IDX]]
        params_list = [params_list[SELECTED_INFER_PARAM_IDX]]

    for ith_threshold, judgement_threshold in enumerate(
            judgement_threshold_array):

        # STAGE 1： 基于召回模型结果，训练故障时间回归模型
        # =======================

        # 准备全局模型训练结果
        # -------------------
        curr_fold = 0
        valid_auc_list, valid_custom_list = [], []
        valid_df_list = []

        # 生成切分的folds
        print("\n[INFO][{}/{}] {} Threshold: {:.10f}, NO: {}".format(
            ith_threshold + 1, len(judgement_threshold_array),
            str(datetime.now())[:-4], judgement_threshold,
            ith_threshold))
        print("--------------------------------")
        time_series_cv_range = yield_train_valid_range()

        for train_range, valid_range in time_series_cv_range:

            train_bool = (train_meta_df["global_day"].values >= train_range[0]) & \
                (train_meta_df["global_day"].values < train_range[1])
            valid_bool = (train_meta_df["global_day"].values >= valid_range[0]) & \
                (train_meta_df["global_day"].values < valid_range[1])

            train_idx = np.arange(0, len(train_meta_df))[train_bool]
            valid_idx = np.arange(0, len(train_meta_df))[valid_bool]

            # STEP 1: 依据训练与验证的bool索引, 构建训练与验证对应的数据集
            # -------------------
            X_train = train_feats[train_idx]
            X_valid = train_feats[valid_idx]

            # STEP 2: 依据训练与验证的bool索引, 构建训练与验证对应标签,
            #         并决定标签的策略(此处采用回归策略)
            # -------------------
            feat_to_predict = "days_to_failure"
            max_threshold = MAX_DAY2FAULT

            # judgement_threshold用来判断是否为故障样本
            # 小于等于judgement_threshold为标签1
            y_train = train_meta_df[feat_to_predict].iloc[train_idx].values
            y_valid = train_meta_df[feat_to_predict].iloc[valid_idx].values

            y_train[np.isnan(y_train)] = max_threshold
            y_valid[np.isnan(y_valid)] = max_threshold
            y_train[y_train >= max_threshold] = max_threshold
            y_valid[y_valid >= max_threshold] = max_threshold

            y_train_label = (y_train < max_threshold).astype(int)
            y_valid_label = (y_valid < max_threshold).astype(int)

            valid_min_date, valid_max_date = \
                np.min(train_group_ids[valid_idx]), np.max(train_group_ids[valid_idx])
            train_label_counting = \
                np.bincount(y_train_label) / len(y_train_label)
            valid_label_counting = \
                np.bincount(y_valid_label) / len(y_valid_label)

            # 用于validation的真实failure tag数据字典
            failure_tag_valid = failure_tag[
                (failure_tag["global_day"] >= valid_range[0]) & \
                (failure_tag["global_day"] < valid_range[1])]
            failure_tag_valid.reset_index(drop=True, inplace=True)

            failure_tag_valid_dict = {}
            for idx in range(len(failure_tag_valid)):
                sn_number = failure_tag_valid.iloc[idx]["serial_number"]
                failure_tag_valid_dict[sn_number] = \
                    np.array([failure_tag_valid.iloc[idx]["min_range"],
                              failure_tag_valid.iloc[idx]["max_range"]],
                              dtype=np.float64)

            # 用于早停的validation的相关参数
            valid_sn_array = \
                train_meta_df["serial_number"].values[valid_idx]
            valid_time_stamp_array = \
                train_meta_df["collect_time_bin_right_edge"].values[valid_idx]
            eval_fcn = partial(
                custom_score_metric,
                sn_array=valid_sn_array,
                time_stamp_array=valid_time_stamp_array,
                custom_threshold=judgement_threshold,
                true_threshold=max_threshold,
                failure_tag_dict=failure_tag_valid_dict)

            # STEP 3: 依据不同参数，训练多组XGBoost模型，并生成
            # train, valid的预测概率
            # -------------------
            custom_f1_metric_list = []

            y_train_pred_list, y_valid_pred_list, valid_iters = [], [], []
            for n_model in range(N_MODELS_PER_FOLD):
                # 检查是否有GPU, 若有GPU则采用GPU构建XGB
                xgb_params = params_list[ith_threshold][n_model]

                # 若有GPU，则采用GPU进行训练
                if gpus:
                    xgb_params["tree_method"] = "gpu_hist"
                    xgb_params["gpu_id"] = GPU_ID

                # 没有valid数据，定轮次进行训练
                for i in np.arange(train_range[0], train_range[1]):
                    if i in np.arange(valid_range[0], valid_range[1]):
                        xgb_params["n_estimators"] = 600
                        is_validation_valid = False
                    else:
                        is_validation_valid = True

                # 开始训练
                xgb_reg = xgb.XGBRegressor(**xgb_params)
                xgb_reg.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    early_stopping_rounds=500,
                    eval_metric=eval_fcn,
                    verbose=False)

                # 生成train与valid上的预测结果并保存
                y_train_pred_list.append(xgb_reg.predict(
                    X_train, ntree_limit=xgb_reg.best_iteration+1))

                y_valid_pred = xgb_reg.predict(
                    X_valid, ntree_limit=xgb_reg.best_iteration+1)
                y_valid_pred_list.append(y_valid_pred)
                valid_iters.append(xgb_reg.best_iteration+1)
                custom_f1_metric_list.append(
                    xgb_reg.evals_result()["validation_0"]["custom_f1"][xgb_reg.best_iteration])

                # 保存预测模型
                TIME2FAULT_JUDGE_MODELS.append(xgb_reg)

            # STEP 5: 评估validation上的结果
            # -------------------
            y_valid_pred = np.mean(y_valid_pred_list, axis=0)
            y_valid_pred = y_valid_pred.clip(0, max_threshold)
            y_valid_pred_proba = (max_threshold - y_valid_pred) / max_threshold
            y_valid_pred_label = \
                (y_valid_pred <= judgement_threshold).astype(int)

            y_valid_df = train_meta_df.iloc[valid_idx].reset_index(drop=True)
            y_valid_df["y_pred_proba"], y_valid_df["y_true"] = \
                y_valid_pred_proba, y_valid_label
            y_valid_df["y_pred_label"] = y_valid_pred_label

            # STAGE 2: 预测validation数据的time2fault时间
            # =======================

            # 归一化数据，并获取预测概率
            X_valid_scaled = time2fault_scaler.transform(X_valid)
            X_valid_pred_time2fault = time2fault_pred_model.predict(
                X_valid_scaled)

            # 依据阈值，获取预测标签
            valid_pred_label = []
            for i in range(X_valid_pred_time2fault.shape[1]):
                valid_pred_label.append(
                    (X_valid_pred_time2fault[:, i] >= time2fault_threshold_list[i]).reshape(-1, 1))
            valid_pred_label = np.hstack(valid_pred_label).astype(int)

            # 基于基本指标，infer可能的故障时间
            valid_time2fault = njit_infer_time2fault(
                valid_pred_label,
                min_fault_minutes=MIN_FAULT_MINUTES,
                gap_minutes=GAP_MINUTES)
            y_valid_df["y_pred_time2fault"] = \
                valid_time2fault

            # STAGE 3: 评估并打印CV结果
            # =======================

            # 全量数据定轮次训练时，valid数据无效
            if is_validation_valid:
                valid_mae = mean_absolute_error(y_valid, y_valid_pred)
                valid_auc = roc_auc_score(
                    y_valid_label.reshape(-1, 1),
                    y_valid_pred_proba.reshape(-1, 1))
                valid_custom, valid_precision, valid_recall, valid_fault_precent = \
                    make_df_evaluation(y_valid_df)
                custom_f1_metric = -np.mean(custom_f1_metric_list)
            else:
                valid_mae, valid_auc = np.nan, np.nan
                valid_custom, valid_precision, valid_recall, valid_fault_precent = \
                    np.nan, np.nan, np.nan, np.nan
                custom_f1_metric = np.nan

            print("[{}] train: {}--{}, valid: {}--{}, avg_iters: {:.2f}, std_iters: {:.2f}".format(
                curr_fold,
                np.min(train_group_ids[train_idx]),
                np.max(train_group_ids[train_idx]),
                valid_min_date, valid_max_date,
                np.mean(valid_iters),
                np.std(valid_iters)))
            print("[{}] valid mae: {:.5f}, auc: {:.5f}, c_f1: {:.5f}".format(
                curr_fold, valid_mae, valid_auc, custom_f1_metric))
            print("[{}] valid custom: {:.5f}, precision: {:.5f},"
                  "recall: {:.5f}, fault_precent: {:.3f}%".format(
                curr_fold, valid_custom, valid_precision,
                valid_recall, valid_fault_precent * 100))
            print("[{}] train: {}, valid: {}".format(
                curr_fold,
                train_label_counting,
                valid_label_counting))
            print("*****************")
            curr_fold += 1

            # 模型与参数保存
            valid_auc_list.append(valid_auc)
            valid_custom_list.append(valid_custom)
            valid_df_list.append([y_valid_df, y_valid_pred_list])

        print("[Average] valid auc: {:.5f}, custom: {:.5f}".format(
            np.nanmean(valid_auc_list),
            np.nanmean(valid_custom_list)))
        print("--------------------------------")
        print("[INFO] {} Model training end.".format(str(datetime.now())[:-4]))

        # STAGE 5: 模型保存部分
        # =======================
        if IS_TRAIN_FOR_INFER:
            # 模型保存
            file_processor = LoadSave(dir_name="./models/")

            # 保存TIME2FAULT JUDGE模型
            file_processor.save_data(
                file_name="time2fault_xgb_judge_models.pkl",
                data_file=TIME2FAULT_JUDGE_MODELS)
