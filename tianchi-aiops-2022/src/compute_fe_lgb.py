# -*- coding: utf-8 -*-

# Created on 202202260047
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
针对训练数据与测试数据，构建训练样本。
'''

import argparse
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from numba import njit
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from models.build_stat import (compute_count_feats, compute_count_svd_feats,
                               compute_df_target_encoding, compute_tfidf_feats,
                               compute_tfidf_svd_feats)
from models.build_utils import (compute_max_embedding, compute_mean_embedding,
                                compute_min_embedding, compute_sif_embedding)
from models.build_word2vec import compute_corpus_embedding
from utils.io_utils import (LoadSave, extract_meta_one_record,
                            process_one_record)
from utils.logger import get_datetime, get_logger

pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 300)


def load_train_test_data():
    '''从本地载入训练数据'''
    file_handler = LoadSave(dir_name='./cached_data/')
    train_df_list = file_handler.load_data(file_name='train_df_list.pkl')
    test_df_list = file_handler.load_data(file_name='test_df_list.pkl')

    return train_df_list, test_df_list


def create_df_corpus_joint(meta_dict):
    '''清洗单个sn的DataFrame中的文本信息，加入server_model用于预训练'''
    if 'server_model' in meta_dict['sel_log'].columns:
        meta_dict['sel_log']['msg'] = meta_dict['sel_log']['msg'] + ' | ' + meta_dict['sel_log']['server_model']
    corpus_tmp = meta_dict['sel_log']['msg'].apply(process_one_record).values.tolist()

    corpus_joint = []
    for item in corpus_tmp:
        corpus_joint.append(' '.join(item))
    corpus_joint = ' '.join(corpus_joint)

    return corpus_joint


def create_df_corpus_joint_tfidf(meta_dict, is_process_one_record=True):
    '''清洗单个DataFrame中的文本信息'''
    if is_process_one_record:
        corpus_tmp = meta_dict['sel_log']['msg'].apply(process_one_record).values.tolist()
    else:
        corpus_tmp = meta_dict['sel_log']['msg_processed'].values.tolist()

    corpus_joint = []
    for item in corpus_tmp:
        corpus_joint.append(' '.join(item))
    corpus_joint = ' '.join(corpus_joint)

    return corpus_joint


@njit
def njit_compute_window_count(timestamp_array, anchor_timestamp, window_size):
    '''统计给定window_size内的相比anchor_timestamp的样本数目'''
    if len(timestamp_array) == 0:
        return 0
    timestamp_reverse_array = timestamp_array[::-1]

    count = 0
    for i in range(len(timestamp_array)):
        if (anchor_timestamp - timestamp_reverse_array[i]) <= window_size:
            count += 1
        else:
            break

    return count


@njit
def njit_compute_window_sum(
    timestamp_array, feat_val_array, ret_dim, anchor_timestamp, window_size
):
    '''计算给定window_size内的相比anchor_timestamp的事件加和数目'''
    sum_feat_array = np.zeros((ret_dim, ))

    if len(timestamp_array) == 0:
        return sum_feat_array
    timestamp_reverse_array = timestamp_array[::-1]
    feat_val_reverse_array = feat_val_array[::-1]

    for i in range(len(timestamp_array)):
        if (anchor_timestamp - timestamp_reverse_array[i]) <= window_size:
            sum_feat_array[feat_val_reverse_array[i]] += 1
        else:
            break

    return sum_feat_array


def compute_log_template_id(log_text, matcher, template_dict):
    '''使用matcher对log_text所属于的模板种类进行匹配'''
    cluster = matcher.match(log_text)

    if cluster and cluster.cluster_id in template_dict:
        return cluster.cluster_id
    else:
        return -1


def quantile(q):
    def func(array):
        return np.quantile(array, q)
    return func


def parse_opt():
    '''解析命令行参数'''
    parser = argparse.ArgumentParser()

    parser.add_argument('--keep-last-k-days', type=int, default=40)
    parser.add_argument('--tfidf-dim', type=int, default=800) # 800
    parser.add_argument('--tfidf-min-range', type=int, default=1)
    parser.add_argument('--tfidf-max-range', type=int, default=2)
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    # 全局化的参数
    # *******************

    # Parse参数列表
    # ----------
    opt = parse_opt()

    # 全局化参数
    # ----------
    KEEP_LAST_K_DAYS = opt.keep_last_k_days
    TFIDF_MAX_FEATS = opt.tfidf_dim
    NGRAM_RANGE = (opt.tfidf_min_range, opt.tfidf_max_range)
    TASK_NAME = 'fe_lgb'
    MSG_EMBEDDING_FILE_NAME_LIST = [
        'cbow_sg_w2v_joint_80d_16w_dict.pkl',
        'cbow_sg_w2v_split_80d_16w_dict.pkl',
    ]
    TEMPLATE_EMBEDDING_FILE_NAME_LIST = [
        'cbow_sg_w2v_template_32d_16w_dict.pkl'
    ]
    TIMEID_EMBEDDING_FILE_NAME_LIST = [
        'cbow_sg_timeid_w2v_64d_16w_dict.pkl'
    ]
    TIME2FAULT_EMBEDDING_FILE_NAME_LIST = [
        'cbow_sg_time2fault_w2v_32d_16w_dict.pkl'
    ]

    timediff_cut_func = lambda x: np.digitize(
        x, np.arange(0, 7 * 24 * 3600, step=120)
    )
    time2fault_cut_func = lambda x: np.digitize(
        x, np.arange(0, 14 * 24 * 3600, step=600)
    )

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

    # 打印特征工程相关参数
    # ----------
    logger.info(opt)

    # 模板提工具
    # ----------
    file_handler = LoadSave(dir_name='./cached_data/')
    template_miner, template_dict = file_handler.load_data(file_name='log_parser.pkl')

    # 载入原始数据并进行简单处理
    # *******************
    train_df_list, test_df_list = load_train_test_data()

    train_sel_log_df, train_label_df, additional_sel_log_df, train_venus_df, train_crashdump_df = train_df_list
    test_sel_log_df, test_submission_df, test_venus_df, test_crashdump_df = test_df_list

    # 日志数据预处理
    # ----------
    logger.info('Training log preprocessing...')

    total_sel_log_df = pd.concat(
        [train_sel_log_df, test_sel_log_df], axis=0
    )
    total_sel_log_df.reset_index(drop=True, inplace=True)

    # 转unix时间戳，方便计算指标
    total_sel_log_df['time_unix'] = total_sel_log_df['time'].astype(int)
    total_sel_log_df['time_unix'] /= 10**9

    # msg日志切分
    total_sel_log_df['msg_processed'] = total_sel_log_df['msg'].apply(process_one_record)

    total_sel_log_df['msg_processed_a'] = total_sel_log_df['msg_processed'].apply(
        lambda x: x[0] if len(x) > 0 else ''
    )
    total_sel_log_df['msg_processed_b'] = total_sel_log_df['msg_processed'].apply(
        lambda x: x[1] if len(x) >= 2 else ''
    )
    total_sel_log_df['msg_processed_c'] = total_sel_log_df['msg_processed'].apply(
        lambda x: x[2] if len(x) == 3 else x[-1]
    )

    # msg日志元信息抽取
    total_sel_log_df['msg_meta'] = total_sel_log_df['msg'].apply(extract_meta_one_record)
    total_sel_log_df['msg_meta'] = total_sel_log_df['msg_meta'].apply(
        lambda x: {x[0][0]: x[0][1]} if len(x) > 0 else None
    )

    # 时间bin id特征
    total_sel_log_df['time_unix_diff'] = total_sel_log_df.groupby(['sn'])['time_unix'].diff()
    total_sel_log_df['time_unix_diff'] = total_sel_log_df['time_unix_diff'].fillna(0)
    total_sel_log_df['time_unix_diff'] = total_sel_log_df['time_unix_diff'].astype(int)
    total_sel_log_df['time_bin_id'] = timediff_cut_func(total_sel_log_df['time_unix_diff'].values)
    total_sel_log_df['time_bin_id'] = total_sel_log_df['time_bin_id'].apply(str)

    # 模板id预处理
    # ----------

    # 匹配日志模板，并计算模板指标
    total_sel_log_df['template_id'] = total_sel_log_df['msg'].apply(
        compute_log_template_id, matcher=template_miner, template_dict=template_dict
    )

    # 模板id重新编码
    encoder = LabelEncoder()
    total_sel_log_df['template_id'] = encoder.fit_transform(total_sel_log_df['template_id'].values)
    total_sel_log_df['template_id'] = total_sel_log_df['template_id'].apply(str)
    template_count = total_sel_log_df['template_id'].nunique()

    # 转换为字典，方便计算特征
    total_sel_log_dict = list(total_sel_log_df.groupby(['sn']))
    total_sel_log_dict = {
        item[0]: item[1].reset_index(drop=True) for item in tqdm(total_sel_log_dict)
    }

    # label数据预处理
    # ----------
    test_submission_df['label'] = np.nan
    total_label_df = pd.concat([train_label_df, test_submission_df], axis=0)
    total_label_df.reset_index(drop=True, inplace=True)

    # 转unix时间戳，方便计算指标
    total_label_df['fault_time_unix'] = total_label_df['fault_time'].astype(int)
    total_label_df['fault_time_unix'] /= 10**9

    # 利用标签数据，构建标签数据集
    # *******************
    logger.info('Construct log df dict...')

    total_sel_log_list = []
    for i in tqdm(range(len(total_label_df))):
        record = total_label_df.iloc[i]

        # 当前record的元信息
        curr_sn, curr_label = record['sn'], record['label']
        curr_end_time = record['fault_time_unix']
        curr_start_time = record['fault_time_unix'] - KEEP_LAST_K_DAYS * 24 * 3600

        # 获取当前样本的sel日志数据
        curr_sel_log = total_sel_log_dict[curr_sn]
        curr_sel_log = curr_sel_log[
            (curr_sel_log['time_unix'] <= curr_end_time) & (curr_sel_log['time_unix'] > curr_start_time)
        ].reset_index(drop=True)

        curr_sel_log['label'] = curr_label

        # 存储list容器中
        total_sel_log_list.append(
            {
                'sel_log': curr_sel_log,
                'sn': curr_sn,
                'label': curr_label,
                'fault_details': record
            }
        )

    # 构建原始语料信息
    # *******************
    logger.info('Construct log corpus...')

    additional_sel_log_dict = list(
        additional_sel_log_df.groupby(['sn'])
    )
    additional_sel_log_dict = {
        item[0]: item[1].reset_index(drop=True) for item in tqdm(additional_sel_log_dict)
    }

    # 构建训练语料集 + additional语料数据集
    msg_raw_corpus_list = []
    for key in tqdm(total_sel_log_dict.keys()):
        tmp_dict = {'sel_log': total_sel_log_dict[key]}

        msg_raw_corpus_list.append(
            create_df_corpus_joint_tfidf(tmp_dict, is_process_one_record=False)
        )

    for key in tqdm(additional_sel_log_dict.keys()):
        tmp_dict = {'sel_log': additional_sel_log_dict[key]}

        msg_raw_corpus_list.append(
            create_df_corpus_joint_tfidf(tmp_dict, is_process_one_record=True)
        )

    # 构建template id语料数据集
    template_raw_corpus_list = []
    for key in tqdm(total_sel_log_dict.keys()):
        template_raw_corpus_list.append(
            ' '.join(
                total_sel_log_dict[key]['template_id'].values.tolist()
            )
        )

    # 特征工程
    # *******************

    # 全局参数与特征DataFrame
    # ----------
    total_feat_df = pd.DataFrame()

    # 基础指标
    # ----------
    total_feat_df['sn'] = [item['sn'] for item in total_sel_log_list]
    total_feat_df['label'] = [item['label'] for item in total_sel_log_list]
    total_feat_df['fault_time'] = [
        item['fault_details']['fault_time'] for item in total_sel_log_list
    ]
    total_feat_df['fault_time_unix'] = [
        item['fault_details']['fault_time_unix'] for item in total_sel_log_list
    ]

    # 标签特征：过去发生的fault的数目，时间差的统计量
    # ----------
    # # 时间差特征
    # total_label_feats = total_feat_df.groupby(['sn'])['fault_time_unix'].diff()
    # total_feat_df['feat/numeric/time2lastFaultEvent'] = total_label_feats.values

    # # 按故障时间groupby计算同时发生的事件个数
    # total_label_feats = total_feat_df.groupby(['sn', 'fault_time_unix'])['fault_time'].agg('count')
    # total_label_feats = total_label_feats.reset_index().rename({'fault_time': 'feat/numeric/eventcount'}, axis=1)

    # total_feat_df = pd.merge(
    #     total_feat_df, total_label_feats, how='left', on=['sn', 'fault_time_unix']
    # )

    total_feat_df.drop(['fault_time_unix'], axis=1, inplace=True)

    # Template id embedding特征
    # ----------
    logger.info('Template id word2vec feature engineering...')

    # 获取语料
    total_corpus_list = []
    for item in tqdm(total_sel_log_list):
        total_corpus_list.append(
            item['sel_log']['template_id'].values.tolist()
        )

    # 获取word2count
    word2count = {}
    for sentence in total_corpus_list:
        for word in sentence:
            if word not in word2count:
                word2count[word] = 1
            else:
                word2count[word] += 1

    for i, embedding_file_name in enumerate(TEMPLATE_EMBEDDING_FILE_NAME_LIST):

        # 载入Embedding文件
        file_handler = LoadSave(dir_name='./cached_data/')
        w2v_dict = file_handler.load_data(file_name=embedding_file_name)

        w2v_mean_embedding = compute_mean_embedding(
            corpus=total_corpus_list,
            word2vec=w2v_dict,
            embedding_size=w2v_dict['feat_dim']
        )
        w2v_min_embedding = compute_max_embedding(
            corpus=total_corpus_list,
            word2vec=w2v_dict,
            embedding_size=w2v_dict['feat_dim']
        )
        w2v_max_embedding = compute_min_embedding(
            corpus=total_corpus_list,
            word2vec=w2v_dict,
            embedding_size=w2v_dict['feat_dim']
        )
        # w2v_sif_embedding = compute_sif_embedding(
        #     total_corpus_list, w2v_dict, word2count
        # )

        w2v_embedding_mat = np.hstack(
            [
                w2v_mean_embedding,
                w2v_min_embedding,
                w2v_max_embedding
            ]
        )

        # 构建词向量
        n_dim = w2v_embedding_mat.shape[1]
        tmp_feat_df = pd.DataFrame(
            w2v_embedding_mat,
            columns=['feat/numeric/w2v_templateid_v{}_{}'.format(i, j) for j in range(n_dim)]
        )
        total_feat_df = pd.concat(
            [total_feat_df, tmp_feat_df], axis=1
        )

    # Template id tf-idf统计特征工程
    # ----------
    logger.info('Template id tf-idf feature engineering...')
    total_corpus_list = [
        ' '.join(item) for item in total_corpus_list
    ]

    # Most frequent的word features
    _, tfidf_vectorizer = compute_tfidf_feats(
        corpus=total_corpus_list, max_feats=200, ngram_range=(1, 2)
    )
    tfidf_array = tfidf_vectorizer.transform(total_corpus_list).toarray()

    for i in range(tfidf_array.shape[1]):
        total_feat_df['feat/numeric/tfidf_template_id_{}'.format(i)] = tfidf_array[:, i]

    # Msg embedding特征
    # ----------
    logger.info('Msg word2vec feature engineering...')

    # 获取向量表示
    total_corpus_list = []
    for item in tqdm(total_sel_log_list):
        total_corpus_list.append(
            create_df_corpus_joint(item)
        )
    total_corpus_list = [item.split(' ') for item in total_corpus_list]

    # 获取word2count
    word2count = {}
    for sentence in total_corpus_list:
        for word in sentence:
            if word not in word2count:
                word2count[word] = 1
            else:
                word2count[word] += 1

    for i, embedding_file_name in enumerate(MSG_EMBEDDING_FILE_NAME_LIST):

        # 载入Embedding文件
        file_handler = LoadSave(dir_name='./cached_data/')
        w2v_dict = file_handler.load_data(file_name=embedding_file_name)

        w2v_mean_embedding = compute_mean_embedding(
            corpus=total_corpus_list,
            word2vec=w2v_dict,
            embedding_size=w2v_dict['feat_dim']
        )
        w2v_min_embedding = compute_max_embedding(
            corpus=total_corpus_list,
            word2vec=w2v_dict,
            embedding_size=w2v_dict['feat_dim']
        )
        w2v_max_embedding = compute_min_embedding(
            corpus=total_corpus_list,
            word2vec=w2v_dict,
            embedding_size=w2v_dict['feat_dim']
        )
        w2v_sif_embedding = compute_sif_embedding(
            total_corpus_list, w2v_dict, word2count
        )

        w2v_embedding_mat = np.hstack(
            [
                w2v_mean_embedding,
                w2v_min_embedding,
                w2v_max_embedding,
                w2v_sif_embedding
            ]
        )

        # 构建词向量
        n_dim = w2v_embedding_mat.shape[1]
        tmp_feat_df = pd.DataFrame(
            w2v_embedding_mat,
            columns=['feat/numeric/w2v_v{}_{}'.format(i, j) for j in range(n_dim)]
        )
        total_feat_df = pd.concat(
            [total_feat_df, tmp_feat_df], axis=1
        )

    # sel_log时间特征
    # ----------
    def compute_sel_log_time_feats(meta_dict):
        '''依据sel_log计算时间戳的相关特征'''
        time_feats_array = []

        log_time_seq = meta_dict['sel_log']['time_unix'].values
        fault_time = meta_dict['fault_details']['fault_time_unix']

        func_list = [
            np.min, np.max, np.std, np.ptp, np.mean, np.median,
            # quantile(0.9), quantile(0.75), quantile(0.25), quantile(0.1)
        ]

        # 日志发生时间差的统计量
        for n_diff in [1]:
            tmp_array = np.diff(log_time_seq, n=n_diff)
            for func in func_list:
                if len(tmp_array) == 0:
                    time_feats_array.append(np.nan)
                else:
                    time_feats_array.append(func(tmp_array))

        tmp_array = log_time_seq / fault_time
        for func in func_list:
            if len(tmp_array) == 0:
                time_feats_array.append(np.nan)
            else:
                time_feats_array.append(func(tmp_array))

        return time_feats_array

    logger.info('Create sel_log time features...')
    total_time_feats = []
    for item in tqdm(total_sel_log_list):
        total_time_feats.append(
            compute_sel_log_time_feats(item)
        )
    total_time_feats = np.vstack(total_time_feats)

    # 构建日志时序特征
    for i in range(total_time_feats.shape[1]):
        total_feat_df['feat/numeric/sel_time_agg_{}'.format(i)] = total_time_feats[:, i]

    # Msg tf-idf统计特征工程
    # ----------
    logger.info('Msg tf-idf feature engineering...')

    total_corpus_list = []
    for item in tqdm(total_sel_log_list):
        total_corpus_list.append(
            create_df_corpus_joint_tfidf(item, is_process_one_record=False)
        )

    _, tfidf_vectorizer = compute_tfidf_feats(
        corpus=msg_raw_corpus_list, max_feats=TFIDF_MAX_FEATS, ngram_range=NGRAM_RANGE
    )

    tfidf_array = tfidf_vectorizer.transform(total_corpus_list).toarray()

    for i in range(tfidf_array.shape[1]):
        total_feat_df['feat/numeric/tfidf_msg_{}'.format(i)] = tfidf_array[:, i]

    # 保存特征工程的结果
    # *******************
    train_feat_df = total_feat_df[total_feat_df['label'].notnull()].reset_index(drop=True)
    train_feat_df['label'] = train_feat_df['label'].astype(int)

    # 排除A阶段训练数据（label == -1）
    # train_feat_df = train_feat_df.query('label != -1').reset_index(drop=True)

    test_feat_df = total_feat_df[total_feat_df['label'].isnull()].reset_index(drop=True)
    test_feat_df.drop(['label'], axis=1, inplace=True)

    logger.info(
        'train_feat_df shape: {}'.format(train_feat_df.shape)
    )
    logger.info(
        'test_feat_df shape: {}'.format(test_feat_df.shape)
    )

    if 'lgb_feats' not in os.listdir('./cached_data/'):
        os.mkdir('./cached_data/{}'.format('lgb_feats'))

    file_handler = LoadSave(dir_name='./cached_data/lgb_feats')
    file_handler.save_data(
        file_name='lgb_feat_df_w{}_list.pkl'.format(KEEP_LAST_K_DAYS),
        data_file=[train_feat_df, test_feat_df]
    )
