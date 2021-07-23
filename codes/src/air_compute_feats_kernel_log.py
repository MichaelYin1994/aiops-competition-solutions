#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202103311628
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

"""
本模块(air_compute_feats_kernel_log.py)对memory_kernel_log日志数据
进行特征工程。
"""

import gc
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from numba import njit

from utils import (LoadSave, create_time2bin_numba_dict, njit_compute_time2bin,
                   INTERVAL_SECONDS,
                   njit_window_group_count_diff,
                   njit_window_group_sum_diff,
                   njit_window_group_unique_count)

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 2022
np.random.seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings("ignore")
###############################################################################

def feat_engineering_kernel_general(total_log):
    """对于kernel log日志中的一般性的特征的抽取。"""
    key_cols = ["serial_number", "collect_time_bin"]
    feat_cols = ["collect_time", "collect_time_bin_right_edge"]

    # 抽取每个观测窗口最后一条日志的信息作为主键
    total_feat_list = []
    total_feat_df = total_log.groupby(key_cols)[feat_cols].last().reset_index()

    group_array = total_log["serial_number"].values
    time_stamp_array = total_log["collect_time"].values

    # 抽取每一时刻的日志给定过去window size(minutes)下的日志条数的counting特征
    # -------------------
    short_term = [15, 360]
    long_term = [720, 1440, 2160]

    for lag in short_term + long_term:
        tmp_vals = njit_window_group_count_diff(
            group_array,
            time_stamp_array,
            time_window_size=lag)

        # 统计每个观测窗口上的统计量
        total_log["lag_count"] = tmp_vals[0]
        total_log["lag_count_diff"] = tmp_vals[1]

        # 计算观测窗口上的shift特征
        window_feat_df = total_log.groupby(
            key_cols)["lag_count", "lag_count_diff"].agg({
                "lag_count": "max",
                "lag_count_diff": "mean"})
        window_feat_df = window_feat_df.reset_index()

        total_feat_list.append(
            window_feat_df[["lag_count", "lag_count_diff"]].values)

    # 拼接所有特征
    # -------------------
    total_log.drop(["lag_count", "lag_count_diff"], axis=1, inplace=True)

    total_feat_list = np.hstack(total_feat_list)
    total_feat_list = pd.DataFrame(
        total_feat_list,
        columns=["kernel_feat_general_{}".format(i) \
                 for i in range(total_feat_list.shape[1])])
    total_feat_df = pd.concat([total_feat_df, total_feat_list], axis=1)

    return total_feat_df


def feat_engineering_kernel_stat(total_log, total_feat_df=None):
    """对于kernel log日志中的一般性的特征的抽取。"""
    key_cols = ["serial_number", "collect_time_bin"]
    feat_cols = ["collect_time", "collect_time_bin_right_edge"]

    # 抽取每个观测窗口最后一条日志的信息作为主键
    total_feat_list = []
    total_feat_df = total_log.groupby(key_cols)[feat_cols].last().reset_index()

    group_array = total_log["serial_number"].values
    time_stamp_array = total_log["collect_time"].values

    oht_cols = [
        '1_hwerr_f', '1_hwerr_e', '2_hwerr_c', '2_sel', '3_hwerr_n',
        '2_hwerr_s', '3_hwerr_m', '1_hwerr_st', '1_hw_mem_c', '3_hwerr_p',
        '2_hwerr_ce', '3_hwerr_as', '1_ke', '2_hwerr_p', '3_hwerr_kp',
        '1_hwerr_fl', '3_hwerr_r', '_hwerr_cd', '3_sup_mce_note', '3_cmci_sub',
        '3_cmci_det', '3_hwerr_pi', '3_hwerr_o', '3_hwerr_mce_l']
    feat_array = total_log[oht_cols].values

    # 抽取kernel log的时间序列滞后Window sum counting统计特征
    # -------------------
    for window_size in [120, 720, 1440, 2880]:
        sum_feats, sum_diff_feats = njit_window_group_sum_diff(
            group_array=group_array,
            time_stamp_array=time_stamp_array,
            feat_array=feat_array,
            time_window_size=window_size)

        # sum_feats处理，提取窗口统计量
        sum_df = pd.DataFrame(
            sum_feats, columns=["kernel_lag_{}_onehot_sum_{}".format(
                window_size, item) for item in range(sum_feats.shape[1])])
        for col in key_cols:
            sum_df[col] = total_log[col]

        sum_feats = sum_df.groupby(key_cols).agg(
            ["max"]).reset_index(drop=True)
        total_feat_list.append(sum_feats.values)

        # # sum_feat_diff处理，提取窗口统计量
        # sum_diff_df = pd.DataFrame(
        #     sum_diff_feats, columns=["kernel_lag_{}_onehot_diff_{}".format(
        #         window_size, item) for item in range(sum_feats.shape[1])])
        # for col in key_cols:
        #     sum_diff_df[col] = total_log[col]

        # sum_diff_feats = sum_diff_df.groupby(key_cols).agg(
        #     ["mean"]).reset_index(drop=True)
        # total_feat_list.append(sum_diff_feats.values)

    # 拼接所有特征
    # -------------------
    total_feat_list = np.hstack(total_feat_list)
    total_feat_list = pd.DataFrame(
        total_feat_list,
        columns=["kernel_feat_stat_{}".format(i) \
                 for i in range(total_feat_list.shape[1])])
    total_feat_df = pd.concat([total_feat_df, total_feat_list], axis=1)

    return total_feat_df


def kernel_log_preprocessing(total_log=None, time2bin_dict=None,):
    """机器kernel log日志数据预处理。"""
    # serial_number预处理
    total_log["serial_number"] = \
        total_log["serial_number"].apply(lambda x: int(x.split("_")[1]))

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
    total_log.fillna(0, axis=1, inplace=True)

    total_log.drop(["vendor", "manufacturer"], axis=1, inplace=True)
    return total_log


if __name__ == "__main__":
    print("***********************")
    print("[INFO] {} kernel_log FE start...".format(
        str(datetime.now())[:-4]))

    # 生成观测窗口间隔，并进行简单预处理
    # ===============================
    time2bin_dict = create_time2bin_numba_dict()

    # 载入所有数据
    # ===============================
    file_processor = LoadSave(dir_name="./data_tmp/")
    kernel_log = file_processor.load_data(
      file_name="kernel_log.pkl")

    # 对每一台机器的日志数据进行特征工程处理，并将结果保存在list中
    # -------------------

    # STEP 1: 机器日志数据预处理
    print("\n***********************")
    print("[INFO] {} kernel_log processing start...".format(
        str(datetime.now())[:-4]))
    kernel_log = kernel_log_preprocessing(
        kernel_log, time2bin_dict)
    
    print("[INFO] {} kernel_log processing start...".format(
        str(datetime.now())[:-4]))
    print("***********************")

    # STEP 2: 基础特征工程
    IS_FEAT_ENGINEERING_GENERAL = True

    if IS_FEAT_ENGINEERING_GENERAL:
        print("\n***********************")
        print("[INFO] {} kernel_log general FE start...".format(
            str(datetime.now())[:-4]))

        # For real computing
        tmp_feats_df = feat_engineering_kernel_general(kernel_log)
        feat_df = tmp_feats_df[
            ["serial_number", "collect_time_bin",
             "collect_time", "collect_time_bin_right_edge"]]

        file_processor.save_data(
            file_name="total_kernel_general.pkl",
            data_file=tmp_feats_df)
        del tmp_feats_df
        gc.collect()

        print("[INFO] {} kernel_log general FE done...".format(
            str(datetime.now())[:-4]))
        print("***********************")

    # STEP 3: 基础特征工程
    IS_FEAT_ENGINEERING_KERNEL_STAT = True

    if IS_FEAT_ENGINEERING_KERNEL_STAT:
        print("\n***********************")
        print("[INFO] {} kernel_log stat FE start...".format(
            str(datetime.now())[:-4]))

        # For real computing
        tmp_feats_df = feat_engineering_kernel_stat(
            kernel_log, feat_df)

        file_processor.save_data(
            file_name="total_kernel_stat.pkl",
            data_file=tmp_feats_df)
        del tmp_feats_df
        gc.collect()

        print("[INFO] {} kernel_log stat FE done...".format(
            str(datetime.now())[:-4]))
        print("***********************")
