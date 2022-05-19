# -*- coding: utf-8 -*-

# Created on 202203090126
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
训练基于Msg信息CBOW和Skip-Gram的词向量模型。
'''

import argparse
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.build_word2vec import compute_cbow_embedding, compute_sg_embedding
from utils.io_utils import LoadSave, process_one_record
from utils.logger import get_datetime, get_logger


def load_train_test_data():
    '''从本地载入训练数据'''
    file_handler = LoadSave(dir_name='./cached_data/')
    train_df_list = file_handler.load_data(file_name='train_df_list.pkl')
    test_df_list = file_handler.load_data(file_name='test_df_list.pkl')

    return train_df_list, test_df_list


def create_df_corpus_joint(meta_dict):
    '''清洗单个DataFrame中的文本信息，加入server_model用于预训练'''
    if 'server_model' in meta_dict['sel_log'].columns:
        meta_dict['sel_log']['msg'] = meta_dict['sel_log']['msg'] + ' | ' + meta_dict['sel_log']['server_model']

    corpus_tmp = meta_dict['sel_log']['msg'].apply(process_one_record).values.tolist()

    corpus_joint = []
    for item in corpus_tmp:
        corpus_joint.append(' '.join(item))
    corpus_joint = ' '.join(corpus_joint)

    return corpus_joint


def create_df_corpus_split(meta_dict):
    '''清洗单个DataFrame中的文本信息，加入server_model用于预训练'''
    if 'server_model' in meta_dict['sel_log'].columns:
        meta_dict['sel_log']['msg'] = meta_dict['sel_log']['msg'] + ' | ' + meta_dict['sel_log']['server_model']

    corpus_tmp = meta_dict['sel_log']['msg'].apply(process_one_record).values.tolist()
    corpus_split = [' '.join(item) for item in corpus_tmp]

    return corpus_split


def parse_opt():
    '''解析命令行参数'''
    parser = argparse.ArgumentParser()

    parser.add_argument('--cbow-n-iters', type=int, default=12)
    parser.add_argument('--sg-n-iters', type=int, default=8)
    parser.add_argument('--window-size', type=int, default=24)
    parser.add_argument('--embedding-dim', type=int, default=32)
    parser.add_argument('--min-count', type=int, default=2)
    parser.add_argument('--style', type=str, default='split')
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    # 全局化的参数
    # *******************

    # Parse参数列表
    # ----------
    opt = parse_opt()
    TASK_NAME = 'training_msg_w2v'

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
        log_path=os.path.join(LOGGING_PATH, LOGGING_FILENAME)
    )

    # 打印训练参数信息
    # ----------
    logger.info(opt)

    # 载入原始数据
    # *******************
    train_df_list, test_df_list = load_train_test_data()

    train_sel_log_df, train_label_df, additional_sel_log_df, _, _ = train_df_list
    test_sel_log_df, test_submission_df, _, _ = test_df_list

    # 日志数据预处理
    # ----------
    total_sel_log_df = pd.concat(
        [train_sel_log_df, test_sel_log_df], axis=0
    )
    total_sel_log_df.reset_index(drop=True, inplace=True)

    total_sel_log_dict = list(total_sel_log_df.groupby(['sn']))
    total_sel_log_dict = {
        item[0]: item[1].reset_index(drop=True) for item in tqdm(total_sel_log_dict)
    }

    # 构建原始语料信息
    # *******************
    logger.warning('opt.style: {} !'.format(opt.style))
    logger.info('Construct corpus...')

    additional_sel_log_dict = list(
        additional_sel_log_df.groupby(['sn'])
    )
    additional_sel_log_dict = {
        item[0]: item[1].reset_index(drop=True) for item in tqdm(additional_sel_log_dict)
    }

    # 构建训练语料集 + 原始语料数据集
    raw_corpus_list = []
    for key in tqdm(total_sel_log_dict.keys()):
        tmp_dict = {'sel_log': total_sel_log_dict[key]}

        if opt.style == 'joint':
            raw_corpus_list.append(
                create_df_corpus_joint(tmp_dict)
            )
        elif opt.style == 'split':
            raw_corpus_list.extend(
                create_df_corpus_split(tmp_dict)
            )

    for key in tqdm(additional_sel_log_dict.keys()):
        tmp_dict = {'sel_log': additional_sel_log_dict[key]}

        if opt.style == 'joint':
            raw_corpus_list.append(
                create_df_corpus_joint(tmp_dict)
            )
        elif opt.style == 'split':
            raw_corpus_list.extend(
                create_df_corpus_split(tmp_dict)
            )

    raw_corpus_list = [item.split(' ') for item in raw_corpus_list]

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
        min_count=opt.min_count,
        window=opt.window_size,
        vector_size=opt.embedding_dim,
        epochs=opt.cbow_n_iters,
        workers=mp.cpu_count(),
        model_name="cbow_{}".format(opt.embedding_dim)
    )

    # 训练Skip-Gram模型
    sg_model = compute_sg_embedding(
        corpus=raw_corpus_list,
        negative=20,
        is_save_model=False,
        min_count=opt.min_count,
        window=opt.window_size,
        vector_size=opt.embedding_dim,
        epochs=opt.sg_n_iters,
        workers=mp.cpu_count(),
        model_name="sg_{}".format(opt.embedding_dim)
    )

    # 拼接词向量，作为基础词向量
    vocab_list = list(cbow_model.wv.key_to_index.keys())

    total_wordvec_dict = dict()
    total_wordvec_dict['feat_dim'] = int(2 * opt.embedding_dim)

    for word in vocab_list:
        cbow_vec = cbow_model.wv[word]
        sg_vec = sg_model.wv[word]
        # fasttext_vec = fasttext_model.wv[word]
        total_wordvec_dict[word] = np.hstack([cbow_vec, sg_vec])

    # 保存word2vec结果
    # *******************
    file_handler = LoadSave(dir_name='./cached_data/')
    file_handler.save_data(
        file_name='cbow_sg_w2v_{}_{}d_{}w_dict.pkl'.format(
            opt.style.lower(), int(opt.embedding_dim * 2), opt.window_size
        ),
        data_file=total_wordvec_dict
    )
