# -*- coding: utf-8 -*-

# Created on 202203071554
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
本模块(build_stat.py)构造统计特征工程工具。
'''

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def compute_tfidf_feats(
        corpus=None, max_feats=100, ngram_range=None, analyzer='word'
    ):
    '''计算稀疏形式的TF-IDF特征，corpus中sentence = ['I love Beijing']'''
    if ngram_range is None:
        ngram_range = (1, 1)

    # 计算TF-IDF特征
    vectorizer = TfidfVectorizer(
        max_features=max_feats, use_idf=True, smooth_idf=True, norm='l2',
        max_df=1.0, analyzer=analyzer, ngram_range=ngram_range,
        token_pattern=r'(?u)\b\w+\b'
    )
    tfidf_array = vectorizer.fit_transform(corpus)

    return tfidf_array, vectorizer


def compute_tfidf_svd_feats(
        corpus=None, max_feats=100, ngram_range=None, analyzer='word', random_state=2912
    ):
    '''计算svd压缩过的TF-IDF特征，corpus中sentence = ['I love Beijing']'''
    if ngram_range is None:
        ngram_range = (1, 1)

    # 计算TF-IDF特征
    vectorizer = TfidfVectorizer(
        max_features=None, use_idf=True, smooth_idf=True, norm='l2',
        max_df=1.0, analyzer=analyzer, ngram_range=ngram_range,
        token_pattern=r'(?u)\b\w+\b'
    )
    tfidf_array = vectorizer.fit_transform(corpus)

    # 计算SVD降维结果
    svd = TruncatedSVD(
        n_components=max_feats, n_iter=12,
        random_state=random_state, algorithm='randomized'
    )
    tfidf_array = svd.fit_transform(tfidf_array)

    return tfidf_array, vectorizer


def compute_count_feats(
    corpus=None, max_feats=100, ngram_range=None, analyzer='word'
):
    '''计算稀疏形式的counting特征，corpus中的sentence = ['I love Beijing']'''
    if ngram_range is None:
        ngram_range = (1, 1)

    vectorizer = CountVectorizer(
        max_features=max_feats,
        max_df=1.0, analyzer=analyzer,
        ngram_range=ngram_range,
        token_pattern=r'(?u)\b\w+\b'
    )
    count_array = vectorizer.fit_transform(corpus)

    return count_array, vectorizer


def compute_count_svd_feats(
    corpus=None, max_feats=100, ngram_range=None, analyzer='word', random_state=2912
):
    '''计算稀疏形式的counting特征，corpus中的sentence = ['I love Beijing']'''
    if ngram_range is None:
        ngram_range = (1, 1)

    vectorizer = CountVectorizer(
        max_features=None,
        max_df=1.0, analyzer=analyzer,
        ngram_range=ngram_range,
        token_pattern=r'(?u)\b\w+\b'
    )
    count_array = vectorizer.fit_transform(corpus)

    # 计算SVD降维结果
    svd = TruncatedSVD(
        n_components=max_feats, n_iter=12,
        random_state=random_state, algorithm='randomized'
    )
    count_array = svd.fit_transform(count_array)

    return count_array, vectorizer


def compute_df_target_encoding(df, feat_col, target_col, m=1):
    '''计算指定属性的target encoding信息'''
    mean_res = df[target_col].mean()

    tmp_df = df.groupby(feat_col).agg(
        count=(target_col, 'count'),
        in_category_mean=(target_col, np.mean)
    ).reset_index()

    tmp_df['weight'] = tmp_df['count'] / (tmp_df['count'] + m)
    tmp_df['score'] = tmp_df['weight'] *  tmp_df['in_category_mean'] + (1 - tmp_df['weight']) * mean_res

    new_feat_col = 'feat/numeric/{}_score'.format(feat_col)
    tmp_df.rename({'score': new_feat_col}, axis=1, inplace=True)

    return tmp_df[[feat_col, new_feat_col]]
