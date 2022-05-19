#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202103292112
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

"""
本模块（test_create_stream_log.py）读取原始的日志数据，并按照复赛要求以stream
的形式发送指定月份的数据。
"""

import gc
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import load_csv, LoadSave

# 设定全局随机种子，并且屏蔽warnings
warnings.filterwarnings("ignore")
###############################################################################

def convert_csv_to_stream_pkl(
        nrows=None, dir_name=None, is_only_fault_log=False):
    """转换原始的日志数据为*pkl形式的按月存储的log数据"""
    if dir_name is None:
        dir_name="./data_tmp/stream_month_log/"

    # 载入全部历史数据
    # ----------------

    # Address log
    # ----------------
    train_memory_address = load_csv(
        dir_name="./data/", nrows=nrows,
        file_name="memory_sample_address_log_round1_a_train.csv")
    test_log = load_csv(
        dir_name="./data/", nrows=nrows,
        file_name="memory_sample_address_log_round1_b_test.csv")
    train_memory_address = pd.concat(
        [train_memory_address, test_log], axis=0).reset_index(drop=True)

    train_memory_address["collect_time"] = pd.to_datetime(
        train_memory_address["collect_time"], format="%Y/%m/%d %H:%M:%S")
    train_memory_address["month"] = \
        train_memory_address["collect_time"].dt.month
    print("[INFO] {} memory_address_log.csv load finished !".format(
        str(datetime.now())[:-4]))

    # MCE log
    # ----------------
    train_mce = load_csv(
        dir_name="./data/", nrows=nrows,
        file_name="memory_sample_mce_log_round1_a_train.csv")
    test_log = load_csv(
        dir_name="./data/", nrows=nrows,
        file_name="memory_sample_mce_log_round1_b_test.csv")
    train_mce = pd.concat(
        [train_mce, test_log], axis=0).reset_index(drop=True)

    train_mce["collect_time"] = pd.to_datetime(
        train_mce["collect_time"], format="%Y/%m/%d %H:%M:%S")
    train_mce["month"] = \
        train_mce["collect_time"].dt.month
    print("[INFO] {} mce_log.csv load finished !".format(
        str(datetime.now())[:-4]))

    # Kernel log
    # ----------------
    train_kernel = load_csv(
        dir_name="./data/", nrows=nrows,
        file_name="memory_sample_kernel_log_round1_a_train.csv")
    test_log = load_csv(
        dir_name="./data/", nrows=nrows,
        file_name="memory_sample_kernel_log_round1_b_test.csv")
    test_log.drop(["failure_time", "tag"], axis=1, inplace=True)
    train_kernel = pd.concat(
        [train_kernel, test_log], axis=0).reset_index(drop=True)

    train_kernel["collect_time"] = pd.to_datetime(
        train_kernel["collect_time"], format="%Y/%m/%d %H:%M:%S")
    train_kernel["month"] = \
        train_kernel["collect_time"].dt.month
    print("[INFO] {} kernel_log.csv load finished !".format(
        str(datetime.now())[:-4]))

    # Failure tag
    # ----------------
    failure_tag = load_csv(
        dir_name="./data/", nrows=nrows,
        file_name="memory_sample_failure_tag_round1_a_train.csv")

    # 是否使用额外泄漏数据作为7月份验证
    IS_USE_ADDITIONAL_DATA = True
    if IS_USE_ADDITIONAL_DATA:
        test_log_b = load_csv(
            dir_name="./data/", nrows=nrows,
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
    failure_tag = failure_tag[["serial_number", "failure_time"]]

    if is_only_fault_log:
        train_memory_address = pd.merge(
            train_memory_address, failure_tag,
            how="left", on=["serial_number"])
        train_memory_address = \
            train_memory_address[train_memory_address["failure_time"].notnull()]
        train_memory_address.reset_index(drop=True, inplace=True)
        train_memory_address.drop(["failure_time"], axis=1, inplace=True)

        train_mce = pd.merge(
            train_mce, failure_tag,
            how="left", on=["serial_number"])
        train_mce = \
            train_mce[train_mce["failure_time"].notnull()]
        train_mce.reset_index(drop=True, inplace=True)
        train_mce.drop(["failure_time"], axis=1, inplace=True)

        train_kernel = pd.merge(
            train_kernel, failure_tag,
            how="left", on=["serial_number"])
        train_kernel = \
            train_kernel[train_kernel["failure_time"].notnull()]
        train_kernel.reset_index(drop=True, inplace=True)
        train_kernel.drop(["failure_time"], axis=1, inplace=True)

    # 按月划分数据，存储数据到本地
    # ----------------

    # 按月query各个表的DataFrame并保存
    train_log_month_list = []
    for month in [1, 2, 3, 4, 5, 7]:
        train_memory_address_tmp = train_memory_address.query(
            "month == {}".format(month)).reset_index(drop=True)
        train_mce_tmp = train_mce.query(
            "month == {}".format(month)).reset_index(drop=True)
        train_kernel_tmp = train_kernel.query(
            "month == {}".format(month)).reset_index(drop=True)

        train_memory_address_tmp.drop(["month"], axis=1, inplace=True)
        train_mce_tmp.drop(["month"], axis=1, inplace=True)
        train_kernel_tmp.drop(["month"], axis=1, inplace=True)

        train_log_month_list.append([
            train_memory_address_tmp,
            train_mce_tmp,
            train_kernel_tmp])

    # 回收表数据，节约内存
    del train_memory_address, train_mce, train_kernel
    del train_memory_address_tmp, train_mce_tmp, train_kernel_tmp
    gc.collect()

    # 按月按分钟保存*.pkl到本地
    month_unique_days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 7: 31}
    for month, (memory_address, mce, kernel) in \
        enumerate(train_log_month_list):
        month += 1
        if month in month_unique_days:
            total_days_of_month = month_unique_days[month]
        else:
            month += 1

        print("***********************")
        print("[INFO] {} Month {} processing start...".format(
            str(datetime.now())[:-4], month))

        # 组装所有log的字典
        month_data_dict = {}
        for minute in range(0, total_days_of_month*24*60):
            month_data_dict[minute] = {
                "address_log": [],
                "mce_log": [],
                "kernel_log": []}

        # 为DataFrame按分钟进行切分，构成分钟级别的数据标签
        minute_bin_unix_time = np.array(pd.date_range(
            start="2019-0{}-01 00:00:00".format(month),
            end="2019-0{}-{} 23:59:59".format(month, total_days_of_month),
            freq="min").astype(np.int64)) // 10**9

        # 扩大第一个元素与最后一个元素的范围
        # 为了纳入左边界第一个元素和数组右边界最后一组元素
        minute_bin_unix_time[0] = int(minute_bin_unix_time[0] - 1)
        minute_bin_unix_time[-1] = int(minute_bin_unix_time[-1] + 61)
        minute_bin_labels = [i for i in range(len(minute_bin_unix_time)-1)]

        memory_address_minute_label = pd.cut(
            memory_address["collect_time"].astype(np.int64)//10**9,
            bins=minute_bin_unix_time,
            labels=minute_bin_labels)
        mce_minute_label = pd.cut(
            mce["collect_time"].astype(np.int64)//10**9,
            bins=minute_bin_unix_time,
            labels=minute_bin_labels)
        kernel_minute_label = pd.cut(
            kernel["collect_time"].astype(np.int64)//10**9,
            bins=minute_bin_unix_time,
            labels=minute_bin_labels)

        # 按分钟顺序存储到month_data_dict里

        # memory address log
        memory_address["collect_time"] = \
            memory_address["collect_time"].astype(str)
        memory_address_vals_list = memory_address.values.tolist()

        for row_idx, minute_label in tqdm(
            enumerate(memory_address_minute_label),
            total=len(memory_address_minute_label)):
            month_data_dict[minute_label]["address_log"].append(
                memory_address_vals_list[row_idx])

        # mce log
        mce["collect_time"] = \
            mce["collect_time"].astype(str)
        mce_vals_list = mce.values.tolist()

        for row_idx, minute_label in tqdm(
            enumerate(mce_minute_label),
            total=len(mce_minute_label)):
            month_data_dict[minute_label]["mce_log"].append(
                mce_vals_list[row_idx])

        # kernel log
        kernel["collect_time"] = \
            kernel["collect_time"].astype(str)
        kernel_vals_list = kernel.values.tolist()

        for row_idx, minute_label in tqdm(
            enumerate(kernel_minute_label),
            total=len(kernel_minute_label)):
            month_data_dict[minute_label]["kernel_log"].append(
                kernel_vals_list[row_idx])

        # 按月份保存数据到本地
        if is_only_fault_log:
            file_name = "{}_stream_fault_log.pkl".format(month)
        else:
            file_name = "{}_stream_log.pkl".format(month)

        file_processor = LoadSave(dir_name=dir_name)
        file_processor.save_data(
            file_name=file_name, data_file=month_data_dict)
    print("***********************")


class GeneratorStreamLog():
    """日志数据生成器。以分钟为单位返回日志数据"""
    def __init__(
            self, selected_month=None, dir_name=None, is_only_fault_log=False):
        if selected_month is None:
            selected_month = [1]
        self.selected_month = sorted(selected_month)
        self.log_size = 0
        self.month_log_dict = {}

        # 按月载入数据
        file_processor = LoadSave(
            dir_name="./data_tmp/stream_month_log/", verbose=0)

        if is_only_fault_log:
            for month in selected_month:
                self.month_log_dict[month] = file_processor.load_data(
                    file_name="{}_stream_fault_log.pkl".format(month))
                self.log_size += len(self.month_log_dict[month])
        else:
            for month in selected_month:
                self.month_log_dict[month] = file_processor.load_data(
                    file_name="{}_stream_log.pkl".format(month))
                self.log_size += len(self.month_log_dict[month])

    def __len__(self):
        return self.log_size

    def __iter__(self):
        for month in self.selected_month:
            month_log = self.month_log_dict[month]
            minutes_list = sorted(list(month_log.keys()))
    
            for minute in minutes_list:
                yield month_log[minute]


if __name__ == "__main__":
    # 全局化参数
    nrows = None

    # 构造用于实时特征测试的数据集
    convert_csv_to_stream_pkl(nrows=nrows, is_only_fault_log=True)
    convert_csv_to_stream_pkl(nrows=nrows, is_only_fault_log=False)

    # 按分钟返回指定月份的数据
    # generator_log = GeneratorStreamLog(selected_month=[7],
    #                                    is_only_fault_log=False)

    # for minute_log in tqdm(generator_log, total=len(generator_log)):
    #     pass
