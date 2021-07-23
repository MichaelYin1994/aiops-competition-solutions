#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202103310028
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

"""
本模块（air_compute_feats_address_log.py）对memory_address_log日志数据
进行特征工程，特征工程的方式主要分为2种：
- 依据预先定间隔（INTERVAL_SECONDS）切分，生成每个样本所属的观测窗口id，并依据
  观测窗口id进行groupby，抽取统计量；同时记录每个观测窗口的右边界，防止时间穿越。
- 利用双指针的方式，抽取每个样本在每个时刻过去给定时间段内的统计量。

本模块主要抽取了两大类的特征：
- general features：根据memory address log的时间顺序，抽取每个观测窗口上的
  Corrected Error的一般性的counting的与counting shift的特征。
- unique count features：对于每一条日志，依据故障原理统计给定窗口间隔内的
  unique的col与row，或者bankid的个数。
"""

import gc
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from numba import njit

from utils import (LoadSave, create_time2bin_numba_dict, INTERVAL_SECONDS,
                   njit_compute_manufacturer2vec, njit_compute_time2bin,
                   njit_compute_vendor2vec, njit_window_group_count_diff,
                   njit_window_group_unique_count, njit_window_group_std,
                   njit_cut_array_to_bin)

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 1989
np.random.seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings("ignore")
###############################################################################
@njit
def njit_forward_fill_nan_address(feat_vals, nan_val=-1):
    """Forward fill nan_val in feat vals."""
    for i in range(len(feat_vals)):
        if i == 0:
            continue
        if feat_vals[i] == nan_val:
            feat_vals[i] = feat_vals[i-1]
    return feat_vals


def feat_engineering_memory_address_general(total_log=None):
    """对于全部的训练与测试的memory address log数据进行一般性特征工程。"""    
    key_cols = ["serial_number", "collect_time_bin"]
    feat_cols = ["collect_time", "collect_time_bin_right_edge"]

    # 抽取每个观测窗口最后一条日志的信息作为主键
    col_names = [name for name in total_log.columns \
        if "vendor" in name or "manufacturer" in name]

    total_feat_df = total_log.groupby(
        key_cols + col_names)[feat_cols].last().reset_index()
    total_feat_df["collect_time_bin_right_edge"] = \
        total_feat_df["collect_time_bin_right_edge"].astype(int)

    # 抽取观测窗口上的count统计量的基础统计特征
    # -------------------
    total_feat_list = []
    group_array = total_log["serial_number"].values
    time_stamp_array = total_log["collect_time"].values

    # 抽取每一时刻的日志给定过去window size(minutes)下的日志条数的counting特征
    # -------------------
    short_term = [15, 120, 360]
    long_term = [720, 1440, 2880]

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
                "lag_count_diff": "median"})
        window_feat_df = window_feat_df.reset_index()

        for n_shift in [10, 50, 150]:
            total_feat_list.append(window_feat_df.groupby(["serial_number"])[
                "lag_count", "lag_count_diff"].shift(n_shift).values)

        total_feat_list.append(
            window_feat_df[["lag_count", "lag_count_diff"]].values)

    # 给定时间点日志过去持续的unique的天数与小时数
    # -------------------
    feat_array = total_log["global_hour"].values.astype(int)
    for window_size in [1440, 2880]:
        total_log["window_stat"] = njit_window_group_unique_count(
            group_array,
            time_stamp_array,
            feat_array,
            time_window_size=window_size)

        # 统计每个观测窗口上的统计量
        window_feat_df = total_log.groupby(
            key_cols)["window_stat"].agg(["max"])
        total_feat_list.append(window_feat_df.values.reshape(-1, 1))

    # 与上条日志发生的时间差
    # -------------------
    total_gp_df = total_log.groupby(
        ["serial_number", "collect_time"])["collect_time_bin"].first().reset_index()

    # 从Unix时间戳的角度抽取diff与count diff的特征
    for n_shift in [5, 50]:
        # 生成shift特征的名字
        feat_name = "address_general_time2last{}log".format(n_shift)

        # Groupby并shift一个到多个时刻
        total_gp_df[feat_name] = total_gp_df.groupby(
            ["serial_number"])["collect_time"].shift(n_shift)
        total_gp_df[feat_name] = \
            total_gp_df["collect_time"] - total_gp_df[feat_name]

        tmp_df = total_gp_df.groupby(
            key_cols)[feat_name].agg(
                 ["mean", "std", "max"])

        # 保存统计量
        total_feat_list.append(tmp_df.values)

        # Shift features
        tmp_df.reset_index(inplace=True)

        for lag in [10, 50]:
            for name in [name for name in tmp_df.columns if "mean" in name]:
                total_feat_list.append(tmp_df.groupby(
                    ["serial_number"])[name].shift(lag).values.reshape(-1, 1))

    # 拼接所有特征
    # -------------------
    total_log.drop(
        ["lag_count", "lag_count_diff", "window_stat"], axis=1, inplace=True)

    total_feat_list = np.hstack(total_feat_list)
    total_feat_list = pd.DataFrame(
        total_feat_list,
        columns=["address_feat_general_{}".format(i) \
                 for i in range(total_feat_list.shape[1])])
    total_feat_df = pd.concat([total_feat_df, total_feat_list], axis=1)

    return total_feat_df


def feat_engineering_memory_memoryid(total_log, total_feat_df=None):
    """针对Memory id进行特征工程。"""
    key_cols = ["serial_number", "collect_time_bin"]
    feat_cols = ["collect_time", "collect_time_bin_right_edge"]

    # 抽取每个观测窗口最后一条日志的信息作为主键
    if total_feat_df is None:
        total_feat_df = total_log.groupby(
            key_cols)[feat_cols].last().reset_index()

    total_feat_list = []
    group_array = total_log["serial_number"].values
    time_stamp_array = total_log["collect_time"].values

    # 抽取memory_id的时间序列滞后unique count统计特征
    # -------------------
    feat_array = total_log["memory"]
    feat_array = feat_array.astype("category").cat.codes.values

    for window_size in [5, 15, 120, 720, 1440, 2880]:
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
        columns=["address_feat_memoryid_{}".format(i) \
                 for i in range(total_feat_list.shape[1])])
    total_feat_df = pd.concat([total_feat_df, total_feat_list], axis=1)

    return total_feat_df


def feat_engineering_memory_rankid(total_log, total_feat_df=None):
    """针对Rank id进行特征工程。"""
    key_cols = ["serial_number", "collect_time_bin"]
    feat_cols = ["collect_time", "collect_time_bin_right_edge"]

    # 抽取每个观测窗口最后一条日志的信息作为主键
    if total_feat_df is None:
        total_feat_df = total_log.groupby(
            key_cols)[feat_cols].last().reset_index()

    total_feat_list = []
    group_array = total_log["serial_number"].values
    time_stamp_array = total_log["collect_time"].values

    # 抽取rank id的时间序列滞后unique count统计特征
    # -------------------
    feat_array = total_log["memory"] * 100 + \
                 total_log["rankid"] * 10000
    feat_array = feat_array.astype("category").cat.codes.values

    for window_size in [5, 15, 120, 720, 1440, 2880]:
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
        columns=["address_feat_rankid_{}".format(i) \
                 for i in range(total_feat_list.shape[1])])
    total_feat_df = pd.concat([total_feat_df, total_feat_list], axis=1)

    return total_feat_df


def feat_engineering_memory_bankid_row_col(total_log, total_feat_df=None):
    """针对Bank id进行特征工程。"""
    key_cols = ["serial_number", "collect_time_bin"]
    feat_cols = ["collect_time", "collect_time_bin_right_edge"]

    # 抽取每个观测窗口最后一条日志的信息作为主键
    if total_feat_df is None:
        total_feat_df = total_log.groupby(
            key_cols)[feat_cols].last().reset_index()

    total_feat_list = []
    group_array = total_log["serial_number"].values
    time_stamp_array = total_log["collect_time"].values

    # 抽取bank id的时间序列滞后unique count统计特征
    # -------------------
    feat_array = total_log["memory"] * 100 + \
                 total_log["rankid"] * 10000 + \
                 total_log["bankid"] * 1000000
    feat_array = feat_array.astype("category").cat.codes.values

    # 30, 720, 1440, 2160
    for window_size in [5, 15, 120, 720, 1440, 2880]:
        total_log["window_stat"] = njit_window_group_unique_count(
            group_array,
            time_stamp_array,
            feat_array,
            window_size).reshape(-1, 1)

        # 统计每个观测窗口上的统计量
        window_feat_df = total_log.groupby(
            key_cols)["window_stat"].agg(["max"])
        total_feat_list.append(window_feat_df.values)

    # 使用one-pass方法计算指定window内的row与col的均值与方差
    # 一定程度上指示了row与col的空间分布特征
    # -------------------
    total_log["memory_rank_bank"] = feat_array
    total_log_tmp = total_log[
        ["serial_number", "memory_rank_bank",
         "collect_time", "collect_time_bin",
         "row", "col"]].sort_values(
            by=["serial_number", "memory_rank_bank", "collect_time"],
            ascending=[True, True, True])

    group_array = total_log_tmp["serial_number"] / 100000 + \
                  total_log_tmp["memory_rank_bank"]
    group_array = group_array.astype("category").cat.codes.values
    time_stamp_array = total_log_tmp["collect_time"].values

    # 针对column error的散布分析
    for window_size in [5, 30, 360, 1440, 2880]:
        total_log_tmp["window_stat"] = njit_window_group_std(
            group_array,
            time_stamp_array,
            total_log_tmp["col"].values,
            window_size).reshape(-1, 1)

        # 统计每个观测窗口上的统计量
        window_feat_df = total_log_tmp.groupby(
            key_cols)["window_stat"].agg(["max"])
        total_feat_list.append(window_feat_df.values)

    # 针对row error的散布分析
    for window_size in [5, 30, 360, 1440, 2880]:
        total_log_tmp["window_stat"] = njit_window_group_std(
            group_array,
            time_stamp_array,
            total_log_tmp["row"].values,
            window_size).reshape(-1, 1)

        # 统计每个观测窗口上的统计量
        window_feat_df = total_log_tmp.groupby(
            key_cols)["window_stat"].agg(["max"])
        total_feat_list.append(window_feat_df.values)

    # 拼接所有特征
    # -------------------
    total_log.drop(["window_stat"], axis=1, inplace=True)

    total_feat_list = np.hstack(total_feat_list)
    total_feat_list = pd.DataFrame(
        total_feat_list,
        columns=["address_feat_bankid_{}".format(i) \
                 for i in range(total_feat_list.shape[1])])
    total_feat_df = pd.concat([total_feat_df, total_feat_list], axis=1)

    return total_feat_df


def feat_engineering_memory_address_row_col(total_log, total_feat_df=None):
    """针对memory address的row与col特征进行特征工程。"""
    key_cols = ["serial_number", "collect_time_bin"]
    feat_cols = ["collect_time", "collect_time_bin_right_edge"]

    # 抽取每个观测窗口最后一条日志的信息作为主键
    if total_feat_df is None:
        total_feat_df = total_log.groupby(
            key_cols)[feat_cols].last().reset_index()

    total_feat_list = []
    group_array = total_log["serial_number"].values
    time_stamp_array = total_log["collect_time"].values

    # 抽取row and col的给定窗口内的unique counting特征
    # ===================

    # Max unique rows
    # -------------------
    feat_array = total_log["memory"] * 100 + \
                 total_log["rankid"] * 10000 + \
                 total_log["bankid"] * 1000000 + \
                 total_log["row"] / 1000000
    feat_array = feat_array.astype("category").cat.codes.values

    for window_size in [5, 15, 120, 720, 1440, 2880]:
        total_log["window_stat"] = njit_window_group_unique_count(
            group_array,
            time_stamp_array,
            feat_array,
            window_size).reshape(-1, 1)

        # 统计每个观测窗口上的统计量
        window_feat_df = total_log.groupby(
            key_cols)["window_stat"].agg(["max"])
        total_feat_list.append(window_feat_df.values)

    # Max unique cols
    # -------------------
    feat_array = total_log["memory"] * 100 + \
                 total_log["rankid"] * 10000 + \
                 total_log["bankid"] * 1000000 + \
                 total_log["col"] / 10000
    feat_array = feat_array.astype("category").cat.codes.values

    for window_size in [5, 15, 120, 720, 1440, 2880]:
        total_log["window_stat"] = njit_window_group_unique_count(
            group_array,
            time_stamp_array,
            feat_array,
            window_size).reshape(-1, 1)

        # 统计每个观测窗口上的统计量
        window_feat_df = total_log.groupby(
            key_cols)["window_stat"].agg(["max"])
        total_feat_list.append(window_feat_df.values)

    # Max unique cells
    # -------------------
    feat_array = total_log["memory"] * 100 + \
                  total_log["rankid"] * 10000 + \
                  total_log["bankid"] * 1000000 + \
                  (total_log["row"] + total_log["col"] / 10000) / 1000000
    feat_array = feat_array.astype("category").cat.codes.values

    for window_size in [5, 15, 120, 720, 1440, 2880]:
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
    total_log.drop(
        ["window_stat"], axis=1, inplace=True)

    total_feat_list = np.hstack(total_feat_list)
    total_feat_list = pd.DataFrame(
        total_feat_list,
        columns=["address_feat_row_col_{}".format(i) \
                 for i in range(total_feat_list.shape[1])])
    total_feat_df = pd.concat([total_feat_df, total_feat_list], axis=1)

    return total_feat_df


def address_log_preprocessing(
        total_log=None, time2bin_dict=None):
    """机器memory address log日志数据预处理。"""

    # serial_number预处理
    # -------------------
    total_log["serial_number"] = \
        total_log["serial_number"].apply(lambda x: int(x.split("_")[1]))

    # 为原始日志的collect_time按INTERVAL_SECONDS为间隔进行切分
    # 并进行缺失值填补等一般性预处理
    # -------------------
    unix_time_array = total_log["collect_time"].values
    time_feat = njit_compute_time2bin(
        unix_time_array, time2bin_dict, 4)

    time_feat = pd.DataFrame(
        time_feat, columns=[
            "collect_time_bin", "global_day",
            "global_hour", "collect_time_bin_right_edge"])
    total_log = pd.concat([total_log, time_feat], axis=1)

    # 前向填充NaN
    for name in ["memory", "rankid", "bankid", "row", "col"]:
        total_log[name] = njit_forward_fill_nan_address(total_log[name].values)

    # 编码vendor与manufacturer特征
    # -------------------
    vendor2vec = np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]])

    manufacturer2vec = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]])

    vendor_feat = njit_compute_vendor2vec(
        total_log["vendor"].values, vendor2vec)
    manufacturer_feat = njit_compute_manufacturer2vec(
        total_log["manufacturer"].values, manufacturer2vec)

    vendor_feat = pd.DataFrame(
        vendor_feat, columns=["vendor_0", "vendor_1", "vendor_2"])
    manufacturer_feat = pd.DataFrame(
        manufacturer_feat,
        columns=["manufacturer_0", "manufacturer_1",
                 "manufacturer_2", "manufacturer_3"])
    total_log.drop(["vendor", "manufacturer"], axis=1, inplace=True)

    total_log = pd.concat(
        [total_log, vendor_feat, manufacturer_feat], axis=1)

    # 切分row与col的特征到bin上
    # -------------------
    # row_width = 8
    # total_log["row_cutted"] = njit_cut_array_to_bin(
    #     total_log["row"].values, low=0, high=2**17-1,
    #     n_bins=int(2**17/row_width))

    # col_width = 8
    # total_log["col_cutted"] = njit_cut_array_to_bin(
    #     total_log["col"].values, low=0, high=2**10-1,
    #     n_bins=int(2**10/col_width))

    return total_log


if __name__ == "__main__":
    print("***********************")
    print("[INFO] {} address_log FE start...".format(
        str(datetime.now())[:-4]))

    # 生成观测窗口间隔，并进行简单预处理
    # ===============================
    time2bin_dict = create_time2bin_numba_dict()

    # 载入所有数据
    # ===============================
    file_processor = LoadSave(dir_name="./data_tmp/")
    address_log = file_processor.load_data(
      file_name="address_log.pkl")
    # 对每一台机器的日志数据进行特征工程处理
    # -------------------

    # STEP 1: 机器日志数据预处理
    print("\n***********************")
    print("[INFO] {} address_log processing start...".format(
        str(datetime.now())[:-4]))
    address_log = address_log_preprocessing(
        address_log, time2bin_dict)
    print("[INFO] {} address_log processing done...".format(
        str(datetime.now())[:-4]))
    print("***********************")

    # STEP 2: 基础特征工程
    IS_FEAT_ENGINEERING_GENERAL = True

    if IS_FEAT_ENGINEERING_GENERAL:
        print("\n***********************")
        print("[INFO] {} address_log general FE start...".format(
            str(datetime.now())[:-4]))
        tmp_feats_df = feat_engineering_memory_address_general(address_log)
        feat_df = tmp_feats_df[
            ["serial_number", "collect_time_bin",
             "collect_time", "collect_time_bin_right_edge"]]

        file_processor.save_data(
            file_name="total_address_general.pkl",
            data_file=tmp_feats_df)
        del tmp_feats_df
        gc.collect()

        print("[INFO] {} address_log general FE done...".format(
            str(datetime.now())[:-4]))
        print("***********************")

    # STEP 3: memoryid特征工程
    IS_FEAT_ENGINEERING_MEMORYID = True

    if IS_FEAT_ENGINEERING_MEMORYID:
        print("\n***********************")
        print("[INFO] {} address_log memoryid FE start...".format(
            str(datetime.now())[:-4]))

        # For real computing
        tmp_feats_df = feat_engineering_memory_memoryid(
            address_log, feat_df)

        file_processor.save_data(
            file_name="total_address_memoryid.pkl",
            data_file=tmp_feats_df)
        del tmp_feats_df
        gc.collect()

        print("[INFO] {} address_log memoryid FE done...".format(
            str(datetime.now())[:-4]))
        print("***********************")

    # STEP 4: rankid特征工程
    IS_FEAT_ENGINEERING_RANKID = True

    if IS_FEAT_ENGINEERING_RANKID:
        print("\n***********************")
        print("[INFO] {} address_log rankid FE start...".format(
            str(datetime.now())[:-4]))

        # For real computing
        tmp_feats_df = feat_engineering_memory_rankid(
            address_log, feat_df)

        file_processor.save_data(
            file_name="total_address_rankid.pkl",
            data_file=tmp_feats_df)
        del tmp_feats_df
        gc.collect()

        print("[INFO] {} address_log rankid FE done...".format(
            str(datetime.now())[:-4]))
        print("***********************")

    # STEP 5: bankid特征工程
    IS_FEAT_ENGINEERING_BANKID = True

    if IS_FEAT_ENGINEERING_BANKID:
        print("\n***********************")
        print("[INFO] {} address_log bankid FE start...".format(
            str(datetime.now())[:-4]))

        # For real computing
        tmp_feats_df = feat_engineering_memory_bankid_row_col(
            address_log, feat_df)

        file_processor.save_data(
            file_name="total_address_bankid.pkl",
            data_file=tmp_feats_df)
        del tmp_feats_df
        gc.collect()

        print("[INFO] {} address_log bankid FE done...".format(
            str(datetime.now())[:-4]))
        print("***********************")

    # STEP 6: row & col特征工程
    IS_FEAT_ENGINEERING_ROW_AND_COL = True

    if IS_FEAT_ENGINEERING_ROW_AND_COL:
        print("\n***********************")
        print("[INFO] {} address_log row & col FE start...".format(
            str(datetime.now())[:-4]))

        # For real computing
        tmp_feats_df = feat_engineering_memory_address_row_col(
            address_log, None)

        file_processor.save_data(
            file_name="total_address_row_col.pkl",
            data_file=tmp_feats_df)
        del tmp_feats_df
        gc.collect()

        print("[INFO] {} address_log row & col FE done...".format(
            str(datetime.now())[:-4]))
        print("***********************")
