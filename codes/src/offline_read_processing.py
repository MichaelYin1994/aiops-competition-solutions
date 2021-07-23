#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202102041518
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

"""
本模块（read_process.py）处理原始日志数据.
"""

import gc
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from utils import LoadSave, load_csv

warnings.filterwarnings("ignore")
###############################################################################

if __name__ == "__main__":
    # 全局化参数
    # -----------------
    N_ROWS = 50000

    # 处理原始的Failure tag信息
    # -----------------
    print("**********************")
    failure_tag = load_csv(
        dir_name="./data/", nrows=N_ROWS,
        file_name="memory_sample_failure_tag_round1_a_train.csv")

    # 是否使用额外泄漏数据作为7月份验证
    IS_USE_ADDITIONAL_DATA = False

    if IS_USE_ADDITIONAL_DATA:
        test_log_b = load_csv(
            dir_name="./data/", nrows=N_ROWS,
            usecols=["serial_number", "manufacturer",
                     "vendor", "failure_time", "tag"],
            file_name="memory_sample_kernel_log_round1_b_test.csv")
        failure_tag_tmp = test_log_b.groupby(
            ["serial_number", "manufacturer", "vendor"])[
                "failure_time", "tag"].agg("last").reset_index()
        failure_tag_tmp = failure_tag_tmp[
            failure_tag_tmp["failure_time"].notnull()]

        del test_log_b
        gc.collect()

        failure_tag_tmp = failure_tag_tmp[list(failure_tag.columns)]
        failure_tag = pd.concat(
            [failure_tag, failure_tag_tmp], axis=0).reset_index(drop=True)

    failure_tag["failure_time"] = pd.to_datetime(
        failure_tag["failure_time"],
        format="%Y/%m/%d %H:%M:%S")

    # 转换原始Timestamp为Unix Time Stamp
    failure_tag["failure_time_unix"] = \
        failure_tag["failure_time"].astype(int) // 10**9
    failure_tag["failure_time_unix"] = \
        failure_tag["failure_time_unix"].astype(np.int32)

    failure_tag["serial_number"] = \
        failure_tag["serial_number"].apply(lambda x: int(x.split("_")[1]))

    # Global stat to the 2019-01-01 00:00:00
    failure_tag["global_day"] = failure_tag["failure_time"] \
            - pd.to_datetime("2019-01-01 00:00:00")
    failure_tag["global_day"] = failure_tag["global_day"].dt.days
    failure_tag["global_day"] = \
        failure_tag["global_day"].astype(np.int32)

    file_processor = LoadSave(dir_name="./data_tmp/")
    file_processor.save_data(
        file_name="failure_tag.pkl",
        data_file=failure_tag)
    print("[INFO] {} failure_tag.pkl save successed !".format(
        str(datetime.now())[:-4]))
    print("**********************")

    # 载入Memory address log数据，并按照serial_number为单位进行存储
    # ----------------
    print("\n**********************")
    print("[INFO] {} memory_address_log.csv processing...".format(
        str(datetime.now())[:-4]))
    train_memory_address = load_csv(
        dir_name="./data/", nrows=N_ROWS,
        file_name="memory_sample_address_log_round1_a_train.csv")
    test_log = load_csv(
        dir_name="./data/", nrows=N_ROWS,
        file_name="memory_sample_address_log_round1_b_test.csv")
    train_memory_address = pd.concat(
        [train_memory_address, test_log], axis=0).reset_index(drop=True)

    train_memory_address["collect_time"] = pd.to_datetime(
        train_memory_address["collect_time"], format="%Y/%m/%d %H:%M:%S")
    train_memory_address["collect_time"] = \
        train_memory_address["collect_time"].astype(int) // 10**9
    train_memory_address.sort_values(
        by=["serial_number", "collect_time"],
        ascending=[True, True], inplace=True)
    train_memory_address.reset_index(drop=True, inplace=True)

    # 以*.pkl格式存储数据
    file_processor = LoadSave(dir_name="./data_tmp/")
    file_processor.save_data(
        file_name="address_log.pkl",
        data_file=train_memory_address)

    del train_memory_address,
    gc.collect()
    print("**********************")

    # 载入Mce log数据，并按照serial_number为单位进行存储
    # ----------------
    print("\n**********************")
    print("[INFO] {} mce_log.csv processing...".format(
        str(datetime.now())[:-4]))
    train_mce = load_csv(
        dir_name="./data/", nrows=N_ROWS,
        file_name="memory_sample_mce_log_round1_a_train.csv")
    test_log = load_csv(
        dir_name="./data/", nrows=N_ROWS,
        file_name="memory_sample_mce_log_round1_b_test.csv")
    train_mce = pd.concat(
        [train_mce, test_log], axis=0).reset_index(drop=True)

    train_mce["collect_time"] = pd.to_datetime(
        train_mce["collect_time"], format="%Y/%m/%d %H:%M:%S")
    train_mce["collect_time"] = \
        train_mce["collect_time"].astype(int) // 10**9
    train_mce.sort_values(
        by=["serial_number", "collect_time"],
        ascending=[True, True], inplace=True)
    train_mce.reset_index(drop=True, inplace=True)

    # 以*.pkl格式存储数据
    file_processor.save_data(
        file_name="mce_log.pkl",
        data_file=train_mce)

    del train_mce
    gc.collect()
    print("**********************")

    # 载入Kernel log数据，并按照serial_number为单位进行存储
    # ----------------
    print("\n**********************")
    print("[INFO] {} kernel_log.csv processing...".format(
        str(datetime.now())[:-4]))
    file_processor = LoadSave(dir_name="./data_tmp/")

    train_kernel = load_csv(
        dir_name="./data/", nrows=N_ROWS,
        file_name="memory_sample_kernel_log_round1_a_train.csv")
    test_log = load_csv(
        dir_name="./data/", nrows=N_ROWS,
        file_name="memory_sample_kernel_log_round1_b_test.csv")
    test_log.drop(["failure_time", "tag"], axis=1, inplace=True)
    train_kernel = pd.concat(
        [train_kernel, test_log], axis=0).reset_index(drop=True)

    train_kernel["collect_time"] = pd.to_datetime(
        train_kernel["collect_time"], format="%Y/%m/%d %H:%M:%S")
    train_kernel["collect_time"] = \
        train_kernel["collect_time"].astype(int) // 10**9
    train_kernel.sort_values(
        by=["serial_number", "collect_time"],
        ascending=[True, True], inplace=True)
    train_kernel.reset_index(drop=True, inplace=True)

    # 以*.pkl格式存储数据
    file_processor.save_data(
        file_name="kernel_log.pkl",
        data_file=train_kernel)

    del train_kernel
    gc.collect()
    print("**********************")
