# -*- coding: utf-8 -*-

# Created on 202111162245
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
构造Skip-gram与CBoW的Word Embedding模型。
'''

import gc
import multiprocessing as mp
import os
import sys
import warnings
from datetime import datetime

sys.path.append('..')

import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import FastText, word2vec
from utils.io_utils import GensimCallback, LoadSave

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 2022
np.random.seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings('ignore')
sns.set(style='ticks', font_scale=1.2, palette='deep', color_codes=True)
###############################################################################

def compute_fasttext_embedding(
    corpus=None, is_save_model=True, model_name='fasttext_model', **kwargs
    ):
    '''利用gensim的FastText模型训练并保存词向量。语料输入形式为：
        [['1', '2', '3'],
        ...,
        ['10', '23', '65', '9', '34']]
    '''
    print('\n[INFO] {} FastText embedding start.'.format(
        str(datetime.now())[:-7])
    )
    print('*******************')
    model = FastText(corpus, **kwargs)
    print('*******************')
    print('[INFO] {} FastText embedding end. \n'.format(
        str(datetime.now())[:-7])
    )

    # 保存Embedding模型
    # ----------
    file_processor = LoadSave(
        dir_name='./pretraining_models/', verbose=1
    )

    if is_save_model:
        file_processor.save_data(
            file_name='{}.pkl'.format(model_name), data_file=model
        )
    return model


def compute_sg_embedding(
    corpus=None, is_save_model=True, model_name='skip_gram_model', **kwargs
    ):
    '''利用gensim的SKip-Gram模型训练并保存词向量。语料输入形式为：
        [['1', '2', '3'],
        ...,
        ['10', '23', '65', '9', '34']]
    '''
    print('\n[INFO] {} Skip-gram embedding start.'.format(
        str(datetime.now())[:-7])
    )
    print('*******************')
    model = word2vec.Word2Vec(corpus, sg=1,
                              compute_loss=True,
                              callbacks=[GensimCallback(verbose_round=1)],
                              **kwargs)
    print('*******************')
    print('[INFO] {} Skip-gram embedding end. \n'.format(
        str(datetime.now())[:-7])
    )

    # 保存Embedding模型
    # ----------
    file_processor = LoadSave(
        dir_name='./pretraining_models/', verbose=1
    )

    if is_save_model:
        file_processor.save_data(
            file_name='{}.pkl'.format(model_name), data_file=model
        )
    return model


def compute_cbow_embedding(
    corpus=None, is_save_model=True, model_name='cbow_model', **kwargs
    ):
    '''利用gensim的CBOW模型训练并保存词向量。语料输入形式为：
        [['1', '2', '3'],
        ...,
        ['10', '23', '65', '9', '34']]
    '''
    print('\n[INFO] CBOW embedding start at {}'.format(
        str(datetime.now())[:-7])
    )
    print('*******************')
    model = word2vec.Word2Vec(corpus, sg=0,
                              compute_loss=True,
                              callbacks=[GensimCallback(verbose_round=1)],
                              **kwargs)
    print('*******************')
    print('[INFO] CBOW embedding end at {}\n'.format(
        str(datetime.now())[:-7])
    )

    # 保存Embedding模型
    # ----------
    file_processor = LoadSave(
        dir_name='./pretraining_models/', verbose=1
    )

    if is_save_model:
        file_processor.save_data(
            file_name='{}.pkl'.format(model_name), data_file=model
        )
    return model


def compute_corpus_embedding(corpus, word2vec, embedding_size):
    '''将句子转化为embedding vector'''
    embedding_mat = np.zeros((len(corpus), embedding_size))

    for idx, seq in enumerate(corpus):
        seq_vec, word_count = np.zeros((embedding_size, )), 0
        for word in seq:
            if word in word2vec:
                seq_vec += word2vec[word]
                word_count += 1

            if word_count != 0:
                embedding_mat[idx, :] = seq_vec / word_count

    return embedding_mat
