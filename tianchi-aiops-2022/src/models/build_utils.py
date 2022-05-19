# -*- coding: utf-8 -*-

# Created on 202203142001
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
模型构建的辅助方法集合。
'''

import numpy as np
from sklearn.decomposition import TruncatedSVD
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm


def compute_mean_embedding(corpus, word2vec, embedding_size):
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


def compute_max_embedding(corpus, word2vec, embedding_size):
    '''将句子转化为embedding vector'''
    embedding_mat = np.zeros((len(corpus), embedding_size))

    for idx, seq in enumerate(corpus):
        tmp_vec_list = []
        for word in seq:
            if word in word2vec:
                tmp_vec_list.append(word2vec[word])

        if len(tmp_vec_list) != 0:
            embedding_mat[idx, :] = np.max(tmp_vec_list, axis=0)

    return embedding_mat


def compute_min_embedding(corpus, word2vec, embedding_size):
    '''将句子转化为embedding vector'''
    embedding_mat = np.zeros((len(corpus), embedding_size))

    for idx, seq in enumerate(corpus):
        tmp_vec_list = []
        for word in seq:
            if word in word2vec:
                tmp_vec_list.append(word2vec[word])

        if len(tmp_vec_list) != 0:
            embedding_mat[idx, :] = np.min(tmp_vec_list, axis=0)

    return embedding_mat


def compute_sif_embedding(sentences, word2vec, word2count, alpha=1e-3, random_state=42):
    '''计算sentences中每个sentcent的SIF embedding信息，返回sentence embedding matrix，'''
    if sentences is None or word2count is None or word2vec is None:
        raise ValueError('Invalid input files !')
    embedding_dim = len(word2vec[list(word2vec.keys())[1]])

    # Step 0: 计算p(w)与word2pw
    total_word_count = 0
    for word, count in word2count.items():
        total_word_count += count

    word2pw = {}
    for word, count in word2count.items():
        word2pw[word] = alpha / (alpha + count / total_word_count)

    # Step 1: 计算sentence embedding
    sentence_embeddings = []
    for sentence in sentences:

        sentence_vec = np.zeros((embedding_dim, ))
        for word in sentence:
            if word in word2vec and word in word2pw:
                sentence_vec += word2pw[word] * word2vec[word]

        sentence_embeddings.append(
            sentence_vec.reshape(1, embedding_dim)
        )
    sentence_embeddings = np.vstack(sentence_embeddings)

    # Step 2: 计算第一主成分
    svd = TruncatedSVD(
        n_components=1, n_iter=7, random_state=random_state, algorithm='randomized'
    )
    first_component = svd.fit(sentence_embeddings).components_.reshape(1, -1)

    # Step 3: 移除第一主成分
    # https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py#L30
    # https://github.com/oborchers/Fast_Sentence_Embeddings
    sentence_embeddings = sentence_embeddings - sentence_embeddings.dot(
        first_component.transpose()
    ) * first_component

    return sentence_embeddings


def build_embedding_matrix(
    word2idx=None, word2embedding=None,
    max_vocab_size=300, embedding_size=128,
    oov_token=None, verbose=False
    ):
    '''利用idx2embedding，组合重新编码过的word2idx。

    Parameters:
    ----------
    word2idx: {dict-like}
        将词语映射到index的字典。键为词语，值为词语对应的index。
    word2embedding: {array-like or dict-like}
        可按照Index被索引的对象，idx2embedding对应了词语的向量，
        通常是gensim的模型对象。
    embedding_size: {int-like}
        embedding向量的维度。
    max_vocab_size: {int-like}
        词表大小，index大于max_vocab_size的词被认为是OOV词。
    oov_token: {str-like}
        未登录词的Token表示。
    verbose: {bool-like}
        是否打印tqdm信息。

    Return:
    ----------
    embedding_mat: {array-like}
        可根据index在embedding_mat的行上进行索引，获取词向量

    References:
    ----------
    [1] https://www.kaggle.com/christofhenkel/keras-baseline-lstm-attention-5-fold
    [2] https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/97471
    '''
    if word2idx is None or word2embedding is None:
        raise ValueError('Invalid Input Parameters !')
    embedding_mat = np.zeros((max_vocab_size+1, embedding_size))

    for word, idx in tqdm(word2idx.items(), disable=not verbose):
        if idx > max_vocab_size:
            continue

        if word in word2embedding:
            embedding_vec = word2embedding[word]
        else:
            embedding_vec = np.array([1] * embedding_size)

        embedding_mat[idx] = embedding_vec

    return embedding_mat


def build_embedding_sequence(
    train_corpus=None,
    test_corpus=None,
    max_vocab_size=1024,
    max_sequence_length=128,
    word2embedding=None,
    oov_token='UNK'
    ):
    '''利用训练与测试语料，基于embedding_model构建用于神经网络的embedding矩阵。

    Parameters:
    ----------
    train_corpus: {list-like}
        包含训练样本的文本序列。每一个元素为一个list，每一个list为训练集的一条句子。
    test_corpus: {list-like}
        包含测试样本的文本序列。每一个元素为一个list，每一个list为测试集的一条句子。
    max_vocab_size: {int-like}
        仅仅编码词频最大的前max_vocab_size个词汇。
    max_sequence_length: {int-like}
        将每一个句子padding到max_sequence_length长度。
    word2embedding: {indexable object}
        可索引的对象，键为词，值为embedding向量。
    oov_token: {str-like}
        语料中的oov_token。

    Returen:
    ----------
    train_corpus_encoded: {list-like}
        经过编码与补长之后的训练集语料数据。
    test_corpus_encoded: {list-like}
        经过编码与补长之后的测试集语料数据。
    embedding_meta: {dict-like}
        包含embedding_mat的基础信息的字典。
    '''
    try:
        embedding_size = word2embedding['feat_dim']
    except KeyError:
        embedding_size = word2embedding.layer1_size

    # 拼接train与test语料数据，获取总语料
    # *******************
    total_corpus = train_corpus + test_corpus

    # 序列化编码语料数据
    # *******************
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(total_corpus)

    word2idx = tokenizer.word_index
    train_corpus_encoded = tokenizer.texts_to_sequences(train_corpus)
    test_corpus_encoded = tokenizer.texts_to_sequences(test_corpus)

    # 补长训练与测试数据，默认以0进行填补
    train_corpus_encoded = pad_sequences(
        train_corpus_encoded, maxlen=max_sequence_length
    )
    test_corpus_encoded = pad_sequences(
        test_corpus_encoded, maxlen=max_sequence_length
    )

    # 构造预训练的embedding matrix
    # *******************
    embedding_mat = build_embedding_matrix(
        word2idx=word2idx,
        word2embedding=word2embedding,
        max_vocab_size=max_vocab_size,
        embedding_size=embedding_size,
        oov_token=oov_token
    )

    embedding_meta = {}
    embedding_meta['embedding_size'] = embedding_mat.shape[1]
    embedding_meta['max_len'] = max_sequence_length
    embedding_meta['max_vocab'] = max_vocab_size
    embedding_meta['embedding_mat'] = embedding_mat
    embedding_meta['tokenizer'] = tokenizer

    return train_corpus_encoded, test_corpus_encoded, embedding_meta
