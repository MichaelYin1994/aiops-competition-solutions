# -*- coding: utf-8 -*-

# Created on 202203270335
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
抽取日志模板信息。
'''

import argparse
import os

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from tqdm import tqdm

from utils.io_utils import LoadSave
from utils.logger import get_datetime, get_logger


def load_train_test_data():
    '''从本地载入训练数据'''
    file_handler = LoadSave(dir_name='./cached_data/')
    train_df_list = file_handler.load_data(file_name='train_df_list.pkl')
    test_df_list = file_handler.load_data(file_name='test_df_list.pkl')

    return train_df_list, test_df_list


def parse_opt():
    '''解析命令行参数'''
    parser = argparse.ArgumentParser()

    parser.add_argument('--keep-top-n-template', type=int, default=300)
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    # 全局化的参数
    # *******************

    # Parse参数列表
    # ----------
    opt = parse_opt()

    TASK_NAME = 'template_extraction'
    KEEP_TOP_N_TEMPLATE = opt.keep_top_n_template

    # 配置日志格式
    # ----------
    LOGGING_PATH = './logs/'
    LOGGING_FILENAME = '{} {}.log'.format(
        get_datetime(), TASK_NAME
    )

    logger = get_logger(
        logger_name=TASK_NAME,
        is_print_std=True,
        is_send_dingtalk=False,
        is_save_to_disk=False,
        log_path=os.path.join(LOGGING_PATH,
        LOGGING_FILENAME)
    )

    # 模板格式抽取配置参数
    # ----------
    config = TemplateMinerConfig()
    config.load('./drain3.ini')
    config.profiling_enabled = False

    drain_file_name = './cached_data/comp_sellog'
    persistence = FilePersistence(drain_file_name + '.bin')
    template_miner = TemplateMiner(persistence, config=config)

    # 载入原始数据
    # *******************
    train_df_list, test_df_list = load_train_test_data()

    train_sel_log_df, train_label_df, additional_sel_log_df, _, _ = train_df_list
    test_sel_log_df, test_submission_df, _, _ = test_df_list

    # 构建日志模板语料并抽取日志模板
    # *******************

    # 构建语料
    # ----------
    logger.info('Construct log corpus...')
    log_text_list = []
    for df in [train_sel_log_df, test_sel_log_df]:
        log_text_list_tmp = df.groupby(['sn'])['msg'].agg(list).values.tolist()

        for log_text in log_text_list_tmp:
            log_text_list.extend(log_text)

    # 建立模板抽取器
    # ----------
    logger.info('Construct log template miner...')
    for log_text in tqdm(log_text_list):
        template_miner.add_log_message(log_text)
    template_count = len(template_miner.drain.clusters)

    # 依据模板数量，统计模板
    # ----------
    template_dict, size_list = {}, []
    for cluster in template_miner.drain.clusters:
        size_list.append(cluster.size)

    size_list = sorted(size_list, reverse=True)[:KEEP_TOP_N_TEMPLATE]
    min_size = size_list[-1]

    # 保存指定要求的模板
    for cluster in template_miner.drain.clusters:
        if cluster.size >= min_size:
            template_dict[cluster.cluster_id] = cluster.size

    # 保存模板抽取器
    # *******************
    file_hander = LoadSave(dir_name='./cached_data/')
    file_hander.save_data(
        file_name='log_parser.pkl', data_file=[template_miner, template_dict]
    )
