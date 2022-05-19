#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202103311521
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

"""
本模块(air_compute_feats_mce_log.py)对mce_log日志数据
进行特征工程，特征工程的方式主要分为2种：
- 依据预先定间隔（INTERVAL_SECONDS）切分，生成每个样本所属的观测窗口id，并依据
  观测窗口id进行groupby，抽取统计量；同时记录每个观测窗口的右边界，防止时间穿越。
- 利用快慢指针的方式，抽取每个样本在每个时刻过去给定时间段内的统计量。

本模块主要抽取了三大类的特征：
- general features：根据mce log的时间顺序，抽取每个给定时间窗口上的
  Corrected Error日志的一般性的counting的与counting shift的特征。
- unique count features：对于每一条日志，依据故障原理统计给定窗口间隔内的
  unique的transaction的个数或者mca_id的个数。
- One-hot统计量特征：抽取每条日志过去给定时间窗口内的mce_id的counting特征
"""

import gc
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from numba import njit

from utils import (LoadSave, create_time2bin_numba_dict, njit_compute_time2bin,
                   njit_compute_transaction2vec, INTERVAL_SECONDS,
                   njit_window_group_count_diff,
                   njit_window_group_sum_diff,
                   njit_window_group_unique_count)

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 1989
np.random.seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings("ignore")
###############################################################################
@njit
def njit_forward_fill_nan_mce(feat_vals, nan_val=-1):
    """Forward fill nan_val in feat vals."""
    for i in range(len(feat_vals)):
        if i == 0:
            continue
        if feat_vals[i] == nan_val:
            feat_vals[i] = feat_vals[i-1]
        elif feat_vals[i] == nan_val and feat_vals[i-1] == nan_val:
            feat_vals[i] = 0
    return feat_vals


def feat_engineering_mce_general(total_log=None):
    """对于mce log日志中的一般性的特征的抽取。"""
    key_cols = ["serial_number", "collect_time_bin"]
    feat_cols = ["collect_time", "collect_time_bin_right_edge"]

    # 抽取每个观测窗口最后一条日志的信息作为主键
    total_feat_list = []
    total_feat_df = total_log.groupby(key_cols)[feat_cols].last().reset_index()

    group_array = total_log["serial_number"].values
    time_stamp_array = total_log["collect_time"].values

    # 抽取每一时刻的日志给定过去window size(minutes)下的日志条数的counting特征
    # -------------------
    short_term = [15, 120, 360]
    long_term = [720, 1440, 2160]

    for lag in short_term + long_term:
        tmp_vals = njit_window_group_count_diff(
            group_array,
            time_stamp_array,
            time_window_size=lag)

        # 统计每个观测窗口上的统计量
        total_log["lag_count"] = tmp_vals[0]
        total_log["lag_count_diff"] = tmp_vals[1]

        total_feat_list.append(
            total_log.groupby(key_cols)[
                ["lag_count"]].agg(["max"]).values)
        total_feat_list.append(
            total_log.groupby(key_cols)[
                "lag_count_diff"].agg(["mean"]).values)

    # 拼接所有特征
    # -------------------
    total_log.drop(["lag_count", "lag_count_diff"], axis=1, inplace=True)

    total_feat_list = np.hstack(total_feat_list)
    total_feat_list = pd.DataFrame(
        total_feat_list,
        columns=["mce_feat_general_{}".format(i) \
                 for i in range(total_feat_list.shape[1])])
    total_feat_df = pd.concat([total_feat_df, total_feat_list], axis=1)

    return total_feat_df


def feat_engineering_mce_mca_id(total_log, total_feat_df=None):
    """针对mca_id进行特征工程。"""
    key_cols = ["serial_number", "collect_time_bin"]
    feat_cols = ["collect_time", "collect_time_bin_right_edge"]

    # 抽取每个观测窗口最后一条日志的信息作为主键
    if total_feat_df is None:
        total_feat_df = total_log.groupby(
            key_cols)[feat_cols].last().reset_index()

    total_feat_list = []
    group_array = total_log["serial_number"].values
    time_stamp_array = total_log["collect_time"].values

    # 抽取mca_id的时间序列滞后unique count统计特征
    # -------------------
    feat_array = total_log["mca_id"]
    feat_array = feat_array.astype("category").cat.codes.values

    for window_size in [5, 15, 60, 720, 1440, 2880]:
        total_log["window_stat"] = njit_window_group_unique_count(
            group_array,
            time_stamp_array,
            feat_array,
            window_size).reshape(-1, 1)

        # 统计每个观测窗口上的统计量
        window_feat_df = total_log.groupby(
            key_cols)["window_stat"].agg(["max"])
        total_feat_list.append(window_feat_df.values)

    # 拼接所有特征
    # -------------------
    total_log.drop(["window_stat"], axis=1, inplace=True)

    total_feat_list = np.hstack(total_feat_list)
    total_feat_list = pd.DataFrame(
        total_feat_list,
        columns=["mce_feat_mcaid_{}".format(i) \
                 for i in range(total_feat_list.shape[1])])
    total_feat_df = pd.concat([total_feat_df, total_feat_list], axis=1)

    return total_feat_df


def feat_engineering_mce_transactionid(total_log, total_feat_df=None):
    """针对transaction id进行特征工程。"""
    key_cols = ["serial_number", "collect_time_bin"]
    feat_cols = ["collect_time", "collect_time_bin_right_edge"]

    # 抽取每个观测窗口最后一条日志的信息作为主键
    if total_feat_df is None:
        total_feat_df = total_log.groupby(
            key_cols)[feat_cols].last().reset_index()

    total_feat_list = []
    group_array = total_log["serial_number"].values
    time_stamp_array = total_log["collect_time"].values

    # 抽取transaction_id的时间序列滞后unique count统计特征
    # -------------------
    feat_array = total_log["transaction"]
    feat_array = feat_array.astype("category").cat.codes.values

    for window_size in [15, 120, 720, 1440, 2880]:
        total_log["window_stat"] = njit_window_group_unique_count(
            group_array,
            time_stamp_array,
            feat_array,
            window_size).reshape(-1, 1)

        # 统计每个观测窗口上的统计量
        window_feat_df = total_log.groupby(
            key_cols)["window_stat"].agg(["max"])
        total_feat_list.append(window_feat_df.values)

    # 抽取transaction id的时间序列滞后One-hot counting统计特征
    # -------------------
    feat_cols = ["transaction_0", "transaction_1",
                 "transaction_2", "transaction_3"]
    total_feat_df_tmp = total_log.groupby(
        key_cols)[feat_cols].sum().reset_index()
    group_array = total_feat_df_tmp["serial_number"].values
    time_stamp_array = \
        total_feat_df_tmp["collect_time_bin"].values * INTERVAL_SECONDS
    feat_array = total_feat_df_tmp[feat_cols].values

    for window_size in [30, 720, 1440, 2880]:
        sum_feats, sum_diff_feats = njit_window_group_sum_diff(
            group_array=group_array,
            time_stamp_array=time_stamp_array,
            feat_array=feat_array,
            time_window_size=window_size)

        total_feat_list.append(sum_feats)
        total_feat_list.append(sum_diff_feats)

    # 拼接所有特征
    # -------------------
    total_log.drop(["window_stat"], axis=1, inplace=True)

    total_feat_list = np.hstack(total_feat_list)
    total_feat_list = pd.DataFrame(
        total_feat_list,
        columns=["mce_feat_transactionid_{}".format(i) \
                 for i in range(total_feat_list.shape[1])])
    total_feat_df = pd.concat([total_feat_df, total_feat_list], axis=1)

    return total_feat_df


def mce_log_preprocessing(total_log=None, time2bin_dict=None, encoder=None):
    """机器mce log日志数据预处理。"""

    # serial_number预处理
    total_log["serial_number"] = \
        total_log["serial_number"].apply(lambda x: int(x.split("_")[1]))

    # 对mca_id列进行编码
    total_log["mca_id"] = encoder.transform(total_log["mca_id"].values)

    # 为原始日志的collect_time按INTERVAL_SECONDS为间隔进行切分
    # 并进行缺失值填补等一般性预处理
    unix_time_array = total_log["collect_time"].values
    time_feat = njit_compute_time2bin(
        unix_time_array, time2bin_dict, 4)

    time_feat = pd.DataFrame(
        time_feat, columns=["collect_time_bin", "global_day",
                            "global_hour", "collect_time_bin_right_edge"])
    total_log = pd.concat([total_log, time_feat], axis=1)

    # forward fillna
    total_log.fillna(-1, axis=1, inplace=True)
    for name in ["transaction", "mca_id"]:
        total_log[name] = njit_forward_fill_nan_mce(total_log[name].values)

    # 编码transaction特征
    transaction2vec = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]])

    transaction_feat = njit_compute_transaction2vec(
        total_log["transaction"].values, transaction2vec)

    transaction_feat = pd.DataFrame(
        transaction_feat,
        columns=["transaction_0", "transaction_1",
                 "transaction_2", "transaction_3"])
    total_log = pd.concat(
        [total_log, transaction_feat], axis=1)

    total_log.drop(["vendor", "manufacturer"], axis=1, inplace=True)
    return total_log


if __name__ == "__main__":
    print("***********************")
    print("[INFO] {} mce_log FE start...".format(
        str(datetime.now())[:-4]))

    # 生成观测窗口间隔，并进行简单预处理
    # ===============================
    time2bin_dict = create_time2bin_numba_dict()

    file_processor = LoadSave(dir_name="./models/")
    encoder_mceid = file_processor.load_data(
        file_name="encoder_mce_log_mca_id.pkl")

    # 载入所有数据
    # ===============================
    file_processor = LoadSave(dir_name="./data_tmp/")
    mce_log = file_processor.load_data(
      file_name="mce_log.pkl")

    # 对每一台机器的日志数据进行特征工程处理，并将结果保存在list中
    # -------------------

    # STEP 1: 机器日志数据预处理
    print("\n***********************")
    print("[INFO] {} mce_log processing start...".format(
        str(datetime.now())[:-4]))
    mce_log = mce_log_preprocessing(
        mce_log, time2bin_dict, encoder_mceid)
    print("[INFO] {} mce_log processing done...".format(
        str(datetime.now())[:-4]))
    print("***********************")

    # STEP 2: 基础特征工程
    IS_FEAT_ENGINEERING_GENERAL = True

    if IS_FEAT_ENGINEERING_GENERAL:
        print("\n***********************")
        print("[INFO] {} mce_log general FE start...".format(
            str(datetime.now())[:-4]))

        # For real computing
        tmp_feats_df = feat_engineering_mce_general(mce_log)
        feat_df = tmp_feats_df[
            ["serial_number", "collect_time_bin",
             "collect_time", "collect_time_bin_right_edge"]]

        file_processor.save_data(
            file_name="total_mce_general.pkl",
            data_file=tmp_feats_df)
        del tmp_feats_df
        gc.collect()

        print("[INFO] {} mce_log general FE done...".format(
            str(datetime.now())[:-4]))
        print("***********************")

    # STEP 2: mca_id工程
    IS_FEAT_ENGINEERING_MCAID = True

    if IS_FEAT_ENGINEERING_MCAID:
        print("\n***********************")
        print("[INFO] {} mce_log mca_id FE start...".format(
            str(datetime.now())[:-4]))

        # For real computing
        tmp_feats_df = feat_engineering_mce_mca_id(
            mce_log, feat_df)

        file_processor.save_data(
            file_name="total_mce_mcaid.pkl",
            data_file=tmp_feats_df)
        del tmp_feats_df
        gc.collect()

        print("[INFO] {} mce_log mca_id FE done...".format(
            str(datetime.now())[:-4]))
        print("***********************")

    # STEP 2: transaction_id工程
    IS_FEAT_ENGINEERING_TRANSACTIONID = True

    if IS_FEAT_ENGINEERING_TRANSACTIONID:
        print("\n***********************")
        print("[INFO] {} mce_log transaction_id FE start...".format(
            str(datetime.now())[:-4]))

        # For real computing
        tmp_feats_df = feat_engineering_mce_transactionid(
            mce_log, feat_df)

        file_processor.save_data(
            file_name="total_mce_transactionid.pkl",
            data_file=tmp_feats_df)
        del tmp_feats_df
        gc.collect()

        print("[INFO] {} mce_log transaction_id FE done...".format(
            str(datetime.now())[:-4]))
        print("***********************")
