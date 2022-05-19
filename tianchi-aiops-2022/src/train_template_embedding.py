# -*- coding: utf-8 -*-

# Created on 202203282121
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
训练基于Template ID信息CBOW和Skip-Gram的词向量模型。
'''

import argparse
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from models.build_word2vec import compute_cbow_embedding, compute_sg_embedding
from utils.io_utils import LoadSave
from utils.logger import get_datetime, get_logger


def load_train_test_data():
    '''从本地载入训练数据'''
    file_handler = LoadSave(dir_name='./cached_data/')
    train_df_list = file_handler.load_data(file_name='train_df_list.pkl')
    test_df_list = file_handler.load_data(file_name='test_df_list.pkl')

    return train_df_list, test_df_list


def compute_log_template_id(log_text, matcher, template_dict):
    '''使用matcher对log_text所属于的模板种类进行匹配'''
    cluster = matcher.match(log_text)

    if cluster and cluster.cluster_id in template_dict:
        return cluster.cluster_id
    else:
        return -1


def parse_opt():
    '''解析命令行参数'''
    parser = argparse.ArgumentParser()

    parser.add_argument('--window-size', type=int, default=16)
    parser.add_argument('--embedding-dim', type=int, default=16)
    opt = parser.parse_args(args=[])

    return opt


if __name__ == '__main__':
    # 全局化的参数
    # *******************

    # Parse参数列表
    # ----------
    opt = parse_opt()

    # 全局化参数
    # ----------
    EMBEDDING_DIM = opt.embedding_dim
    WINDOW_SIZE = opt.window_size
    TASK_NAME = 'training_template_w2v'

    # 配置日志格式
    # ----------
    LOGGING_PATH = './logs/'
    LOGGING_FILENAME = '{} {}.log'.format(
        get_datetime(), TASK_NAME
    )

    logger = get_logger(
        logger_name=TASK_NAME,
        is_print_std=True, is_send_dingtalk=False, is_save_to_disk=False,
        log_path=os.path.join(LOGGING_PATH, LOGGING_FILENAME)
    )

    # 模板提工具
    # ----------
    file_handler = LoadSave(dir_name='./cached_data/')
    template_miner, template_dict = file_handler.load_data(file_name='log_parser.pkl')

    # 数据载入
    # *******************
    file_handler = LoadSave(dir_name='./cached_data/', verbose=0)
    train_df_list, test_df_list = load_train_test_data()

    train_sel_log_df, train_label_df, additional_sel_log_df, _, _ = train_df_list
    test_sel_log_df, test_submission_df, _, _ = test_df_list

    # 日志数据预处理
    # ----------
    logger.info('Training log preprocessing...')

    total_sel_log_df = pd.concat(
        [train_sel_log_df, test_sel_log_df], axis=0
    )
    total_sel_log_df.reset_index(drop=True, inplace=True)

    # 匹配日志模板，并计算模板指标
    total_sel_log_df['template_id'] = total_sel_log_df['msg'].apply(
        compute_log_template_id,
        matcher=template_miner,
        template_dict=template_dict
    )

    # 构建template id语料
    # ----------
    encoder = LabelEncoder()
    total_sel_log_df['template_id'] = encoder.fit_transform(total_sel_log_df['template_id'].values)
    template_count = total_sel_log_df['template_id'].nunique()

    # 将日志模板序列转换为语料
    total_sel_log_corpus_df = total_sel_log_df.groupby(['sn'])['template_id'].agg(list).reset_index()
    total_sel_log_corpus_df['template_id'] = total_sel_log_corpus_df['template_id'].apply(
        lambda x: [str(item) for item in x]
    )
    raw_corpus_list = total_sel_log_corpus_df['template_id'].values.tolist()

    # 训练Embedding模型
    # *******************

    # 训练模型
    # ----------
    logger.info('Training word embedding model...')

    # 训练CBOW模型
    cbow_model = compute_cbow_embedding(
        corpus=raw_corpus_list,
        negative=20,
        is_save_model=False,
        min_count=2,
        window=WINDOW_SIZE,
        vector_size=EMBEDDING_DIM,
        epochs=25,
        workers=int(mp.cpu_count() / 2),
        model_name="cbow_{}".format(EMBEDDING_DIM)
    )

    # 训练Skip-Gram模型
    sg_model = compute_sg_embedding(
        corpus=raw_corpus_list,
        negative=20,
        is_save_model=False,
        min_count=2,
        window=WINDOW_SIZE,
        vector_size=EMBEDDING_DIM,
        epochs=17,
        workers=int(mp.cpu_count() / 2),
        model_name="sg_{}".format(EMBEDDING_DIM)
    )

    # 拼接CBOW与Skip-Gram词向量，作为基础词向量
    vocab_list = list(cbow_model.wv.key_to_index.keys())

    cbow_sg_wordvec_dict = dict()
    cbow_sg_wordvec_dict['feat_dim'] = int(2 * EMBEDDING_DIM)

    for word in vocab_list:
        cbow_vec = cbow_model.wv[word]
        sg_vec = sg_model.wv[word]
        cbow_sg_wordvec_dict[word] = np.hstack([cbow_vec, sg_vec])

    # 保存word2vec结果
    # *******************
    file_handler = LoadSave(dir_name='./cached_data/')
    file_handler.save_data(
        file_name='cbow_sg_w2v_{}_{}d_{}w_dict.pkl'.format(
            'template', int(EMBEDDING_DIM * 2), WINDOW_SIZE
        ),
        data_file=cbow_sg_wordvec_dict
    )
