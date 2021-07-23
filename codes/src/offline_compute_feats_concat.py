#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202103312003
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

"""
本模块(offline_compute_feats_concat.py)对mce_log特征数据，memory address特征
数据，kernel log特征数据按照观测窗口id和serial_number进行Merge；并制作标签。
"""

import gc
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import LoadSave

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 2022
np.random.seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings("ignore")
###############################################################################
def load_concat_all_addresslog_df():
    """将memory_address的特征载入内存并进行concat，返回concat后的df"""
    file_processor = LoadSave(dir_name="./data_tmp/")
    file_names = ["total_address_general.pkl", "total_address_memoryid.pkl",
                  "total_address_rankid.pkl", "total_address_bankid.pkl",
                  "total_address_row_col.pkl"]
    key_cols = \
        ["serial_number", "collect_time",
         "collect_time_bin", "collect_time_bin_right_edge"]

    total_feats = []
    for i in [0, 1, 2, 3, 4]:
        file_name = file_names[i]

        if len(total_feats) == 0:
            total_feats.append(file_processor.load_data(
                file_name=file_name))
        else:
            total_feats.append(file_processor.load_data(
                file_name=file_name).drop(key_cols, axis=1))

    total_feats = pd.concat(total_feats, axis=1)

    # Save metainfo
    file_processor = LoadSave(
        dir_name="./models/", verbose=0)
    file_processor.save_data(
        file_name="meta_address_log_feats.pkl",
        data_file={"n_samples": len(total_feats),
                   "n_feats": int(total_feats.shape[1] - len(key_cols))})

    return total_feats


def load_concat_all_mcelog_df():
    """将mcelog的特征载入内存并进行concat，返回concat后的df"""
    file_processor = LoadSave(dir_name="./data_tmp/")
    file_names = ["total_mce_general.pkl", "total_mce_mcaid.pkl",
                  "total_mce_transactionid.pkl"]
    key_cols = \
        ["serial_number", "collect_time",
         "collect_time_bin", "collect_time_bin_right_edge"]

    total_feats = []
    for i in [1, 2]:
        file_name = file_names[i]

        if len(total_feats) == 0:
            total_feats.append(file_processor.load_data(
                file_name=file_name))
        else:
            total_feats.append(file_processor.load_data(
                file_name=file_name).drop(key_cols, axis=1))

    total_feats = pd.concat(total_feats, axis=1)

    # Save metainfo
    file_processor = LoadSave(
        dir_name="./models/", verbose=0)
    file_processor.save_data(
        file_name="meta_mce_log_feats.pkl",
        data_file={"n_samples": len(total_feats),
                   "n_feats": int(total_feats.shape[1] - len(key_cols))})

    return total_feats


def load_concat_all_kernellog_df():
    """将kernellog的特征载入内存并进行concat，返回concat后的df"""
    file_processor = LoadSave(dir_name="./data_tmp/")
    file_names = ["total_kernel_general.pkl",
                  "total_kernel_stat.pkl"]
    key_cols = \
        ["serial_number", "collect_time",
         "collect_time_bin", "collect_time_bin_right_edge"]

    total_feats = []
    for i in [1]:
        file_name = file_names[i]

        if len(total_feats) == 0:
            total_feats.append(file_processor.load_data(
                file_name=file_name))
        else:
            total_feats.append(file_processor.load_data(
                file_name=file_name).drop(key_cols, axis=1))

    total_feats = pd.concat(total_feats, axis=1)

    # Save metainfo
    file_processor = LoadSave(
        dir_name="./models/", verbose=0)
    file_processor.save_data(
        file_name="meta_kernel_log_feats.pkl",
        data_file={"n_samples": len(total_feats),
                   "n_feats": int(total_feats.shape[1] - len(key_cols))})

    return total_feats


if __name__ == "__main__":
    # 载入所有数据集
    # ===============================
    file_processor = LoadSave(dir_name="./data_tmp/", verbose=1)
    failure_tag = file_processor.load_data(
        file_name="failure_tag.pkl")

    total_df = load_concat_all_addresslog_df()

    # 多表合体
    # ===============================
    # total_address_log接入total_df信息，回收内存
    # -------------------
    IS_USE_ADDRESS_LOG = False

    if IS_USE_ADDRESS_LOG:
        merge_list= ["serial_number", "collect_time_bin",
                     "collect_time_bin_right_edge"]
        drop_list = ["collect_time"]

        total_address_df = load_concat_all_addresslog_df()
        total_address_df.drop(drop_list, axis=1, inplace=True)
        total_df = pd.merge(total_df, total_address_df,
                            on=merge_list, how="left")

        del total_address_df
        gc.collect()

    # total_mce_log接入total_log信息，回收内存
    # -------------------
    IS_USE_MCE_LOG = True

    if IS_USE_MCE_LOG:
        merge_list= ["serial_number", "collect_time_bin",
                     "collect_time_bin_right_edge"]
        drop_list = ["collect_time"]

        total_mce_df = load_concat_all_mcelog_df()
        total_mce_df.drop(drop_list, axis=1, inplace=True)
        total_df = pd.merge(total_df, total_mce_df,
                            on=merge_list, how="left")

        del total_mce_df
        gc.collect()

    # total_kernel_log接入total_log信息，回收内存
    # -------------------
    IS_USE_KERNEL_LOG = True

    if IS_USE_KERNEL_LOG:
        merge_list= ["serial_number", "collect_time_bin",
                     "collect_time_bin_right_edge"]
        drop_list = ["collect_time"]

        total_kernel_df = load_concat_all_kernellog_df()
        total_kernel_df.drop(drop_list, axis=1, inplace=True)
        total_df = pd.merge(total_df, total_kernel_df,
                            on=merge_list, how="left")

        del total_kernel_df
        gc.collect()

    # 所有np.nan特征统一用0进行填充
    # -------------------
    total_df.fillna(0, inplace=True)

    # 处理failure_tag信息，制作标签
    # ===============================
    failure_label_df = []

    for i in tqdm(range(len(failure_tag))):
        failure_datetime = failure_tag.iloc[i]["failure_time"]

        # 特殊处理, 以分钟为单位生成index数组
        failure_datetime_range = pd.date_range(
            start=failure_datetime + pd.Timedelta(-15, unit="d"),
            end=failure_datetime, freq="min")

        failure_datetime_range = pd.DataFrame(
            failure_datetime_range,
            columns=["failure_time"])

        # Meta information
        failure_datetime_range["serial_number"] = \
            failure_tag.iloc[i]["serial_number"]

        # 暂时不用tag
        # failure_datetime_range["tag"] = failure_tag.iloc[i]["tag"]
        failure_datetime_range["days_to_failure"] = \
            failure_datetime - failure_datetime_range["failure_time"]
        failure_datetime_range["days_to_failure"] = \
            failure_datetime_range["days_to_failure"] / np.timedelta64(1, "h") / 24

        # https://stackoverflow.com/questions/36615565/is-it-possible-to-convert-timedelta-into-hours
        failure_datetime_range["hours_to_failure"] = \
            failure_datetime - failure_datetime_range["failure_time"]
        failure_datetime_range["hours_to_failure"] = \
            failure_datetime_range["hours_to_failure"] / np.timedelta64(1, "h")

        failure_datetime_range["month"] = \
            failure_datetime_range["failure_time"].dt.month
        failure_datetime_range["day"] = \
            failure_datetime_range["failure_time"].dt.day
        failure_datetime_range["hour"] = \
            failure_datetime_range["failure_time"].dt.hour
        failure_datetime_range["minute"] = \
            failure_datetime_range["failure_time"].dt.minute
        failure_datetime_range.drop(["failure_time"], axis=1, inplace=True)

        failure_label_df.append(failure_datetime_range)
    failure_label_df = pd.concat(failure_label_df, axis=0)

    # 合并failure_label_df与total_address_log数据
    # -------------------
    total_df["collect_time_dt"] = \
        pd.to_datetime(total_df["collect_time"].values, unit="s")
    total_df["month"] = \
        total_df["collect_time_dt"].dt.month
    total_df["day"] = \
        total_df["collect_time_dt"].dt.day
    total_df["hour"] = \
        total_df["collect_time_dt"].dt.hour
    total_df["minute"] = \
        total_df["collect_time_dt"].dt.minute

    total_df = pd.merge(
        total_df, failure_label_df, how="left",
        on=["serial_number", "month", "day",
            "hour", "minute"])

    total_df["global_day"] = total_df["collect_time_dt"] \
            - pd.to_datetime("2019-01-01 00:00:00")
    total_df["global_day"] = total_df["global_day"].dt.days
    total_df["global_day"] = \
        total_df["global_day"].astype(np.int32)

    # 保存构造好的特征数据
    # ===============================
    drop_list = ["day", "hour", "minute", "collect_time_dt",
                 "collect_time", "collect_time_bin"]
    total_df.drop(drop_list, axis=1, inplace=True)

    # reduce_memory = ReduceMemoryUsage(
    #     data_table=total_df, verbose=False)
    # total_df = reduce_memory.reduce_memory_usage()

    file_processor.save_data(
        file_name="xgb_total_feats_df.pkl", data_file=total_df)
    print(total_df.shape)
