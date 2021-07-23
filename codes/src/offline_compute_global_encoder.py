#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202103301427
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

"""
读入全部训练*.csv数据，获取对于str对象的编码encoder，并存储于Models
文件夹当中。
"""

import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import INTERVAL_SECONDS, CategoricalEncoder, LoadSave, load_csv

warnings.filterwarnings("ignore")
###############################################################################

if __name__ == "__main__":
    # 全局化参数
    N_ROWS = None

    # 载入全部历史数据
    # ----------------
    train_mce = load_csv(
        dir_name="./data/", nrows=N_ROWS,
        file_name="memory_sample_mce_log_round1_a_train.csv")
    test_log = load_csv(
        dir_name="./data/", nrows=N_ROWS,
        file_name="memory_sample_mce_log_round1_b_test.csv")
    train_mce = pd.concat(
        [train_mce, test_log], axis=0).reset_index(drop=True)

    file_processor = LoadSave(dir_name="./models/")

    # Encoding mce_log
    # ----------------
    feat2encode = ["mca_id"]

    for name in feat2encode:
        encoder_name = "encoder_mce_log_{}.pkl".format(name)

        # 构建特征的encoder并保存到本地
        encoder = CategoricalEncoder()
        encoder.fit(raw_data=train_mce[name].values)

        file_processor.save_data(
            file_name=encoder_name, data_file=encoder)
