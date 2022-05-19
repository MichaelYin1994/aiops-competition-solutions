# -*- coding: utf-8 -*-

# Created on 202110060546
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
模型训练数据管道相关API。
'''

import os
import pickle
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

warnings.filterwarnings('ignore')
###############################################################################


def check_hex(text):
    '''检查输入text是否为hex字符串，是返回1,否则返回0'''
    if len(text) <= 1:
        return 0

    if text.startswith('0'):
        return 1
    else:
        return 0


# port dimm cpu nvmessd ps
def process_one_record(msg_text):
    '''处理单条Error Log Message信息'''
    if msg_text is None or len(msg_text) == 0:
        return ''

    msg_token_list = []
    msg_text_splitted = msg_text.strip().split('|')

    for msg_segment in msg_text_splitted:
        msg_segment = msg_segment.strip().lower().replace('_', ' ')
        tmp_token_list = re.split('\W+', msg_segment)

        # 滤除空字符
        tmp_token_list = [
            item.strip() for item in tmp_token_list if len(item) != 0
        ]

        # 替换hex字符串
        func = lambda x: 'hex' if check_hex(x) else x
        tmp_token_list = [
            func(item) for item in tmp_token_list
        ]

        # 规则修正关键字符串信息
        for i in range(len(tmp_token_list)):
            curr_word = tmp_token_list[i]

            if 'port' in curr_word:
                tmp_token_list[i] = 'port'
            elif 'dimm' in curr_word:
                tmp_token_list[i] = 'dimm'

        msg_token_list.append(
            ' '.join(tmp_token_list)
        )

    return msg_token_list


def extract_meta_one_record(msg_text):
    '''抽取单条Msg Text中的元信息'''
    if msg_text is None or len(msg_text) == 0:
        return ''

    msg_meta_list = []
    msg_text_splitted = msg_text.strip().split('|')

    for msg_segment in msg_text_splitted:
        msg_segment = msg_segment.strip().lower().replace('_', ' ')
        tmp_token_list = re.split('\W+', msg_segment)

        # 滤除空字符
        tmp_token_list = [
            item.strip() for item in tmp_token_list if len(item) != 0
        ]

        # 抽取addr元信息
        for i, token in enumerate(tmp_token_list):
            if check_hex(token):
                msg_meta_list.append(
                    [' '.join(tmp_token_list[max(i-2, 0):i]), token]
                )
            elif 'cpu' in token and len(token) > 3:
                msg_meta_list.append(
                    [' '.join(tmp_token_list[max(i-2, 0):i]), token]
                )

    return msg_meta_list


class LogTextCnnDataGenerator(tf.keras.utils.Sequence):
    '''日志数据神经网络训练/测试数据的生成器'''
    def __init__(
        self, log_corpus, template_corpus, time_seq, dense_feats, targets,
        log_tokenizer, template_tokenizer,
        log_corpus_maxlen=512, template_corpus_maxlen=512, **kwargs
        ):

        self.log_corpus = log_corpus
        self.template_corpus = template_corpus
        self.time_seq = time_seq
        self.dense_feats = dense_feats
        self.targets = targets
        self.indexes = np.arange(len(log_corpus))

        self.log_tokenizer = log_tokenizer
        self.template_tokenizer = template_tokenizer

        self.log_corpus_maxlen = log_corpus_maxlen
        self.template_corpus_maxlen = template_corpus_maxlen

        self.batch_size = kwargs.pop('batch_size', 16)
        self.is_train = kwargs.pop('is_train', True)
        self.shuffle_on_epoch_end = kwargs.pop('shuffle_on_epoch_end', False)

    def __len__(self):
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        # 按照indexes取序号（注意除不尽的batch end）
        final_batch_id = len(self)

        if index == final_batch_id:
            indexes = self.indexes[
                int(index*self.batch_size):
            ]
        else:
            indexes = self.indexes[
                int(index*self.batch_size):int((index+1)*self.batch_size)
            ]

        # 按照序号取数据进行编码
        batch_labels = []
        batch_sentences, batch_templates = [], []
        batch_time_seq, batch_dense_feats = [], []

        for i in indexes:
            # log corpus
            sentence = ' '.join(self.log_corpus[i]).split(' ')
            batch_sentences.append(sentence)

            # template corpus
            templates = self.template_corpus[i]
            batch_templates.append(templates)

            # time sequences
            time_seq = np.zeros((self.template_corpus_maxlen, 2))
            curr_time_seq = self.time_seq[i]
            time_seq[:len(curr_time_seq), :] = curr_time_seq[:self.template_corpus_maxlen, :]
            batch_time_seq.append(time_seq)

            # dense features
            dense_feats = self.dense_feats[i, :]
            batch_dense_feats.append(dense_feats.reshape(1, -1))

            if self.is_train:
                batch_labels.append(self.targets[i, :].reshape(1, -1))

        # log corpus
        batch_token_ids = self.log_tokenizer.texts_to_sequences(batch_templates)
        batch_token_ids = tf.keras.preprocessing.sequence.pad_sequences(
            batch_token_ids, maxlen=self.log_corpus_maxlen,
            padding='post', truncating='post',
        )

        # template corpus
        batch_template_ids = self.template_tokenizer.texts_to_sequences(batch_sentences)
        batch_template_ids = tf.keras.preprocessing.sequence.pad_sequences(
            batch_template_ids, maxlen=self.template_corpus_maxlen,
            padding='post', truncating='post',
        )

        # time sequences
        batch_time_seq = np.concatenate(
            [np.expand_dims(item, 0) for item in batch_time_seq]
        )
        batch_dense_feats = np.concatenate(batch_dense_feats, axis=0)

        batch_encoded_res = [
            batch_token_ids, batch_template_ids, batch_time_seq, batch_dense_feats
        ]

        # 返回编码后的numpy array
        if self.is_train:
            batch_labels = np.concatenate(batch_labels, axis=0).astype(int)
            return batch_encoded_res, batch_labels
        else:
            return batch_encoded_res

    def on_epoch_end(self):
        if self.shuffle_on_epoch_end:
            np.random.shuffle(self.indexes)

            # for i in range(len(self.corpus)):
            #     tmp_idx = np.arange(len(self.corpus[i]))
            #     np.random.shuffle(tmp_idx)
            #     self.corpus[i] = [self.corpus[i][j] for j in tmp_idx]
            #     # self.time_seq[i] = self.time_seq[i][]


def load_from_csv(dir_name, file_name, dataframe, **kwargs):
    '''从dir_name路径读取file_name的文件'''
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError('Input data must be a DataFrame !')
    if not isinstance(dir_name, str) or not isinstance(file_name, str):
        raise TypeError('Input dir_name and file_name must be str type !')
    if not file_name.endswith('.csv'):
        raise ValueError('Input file_name must end with *.csv !')

    full_name = os.path.join(dir_name, file_name)
    return pd.read_csv(full_name, **kwargs)


class GensimCallback(CallbackAny2Vec):
    '''计算每一个Epoch的词向量训练损失的回调函数。

    Attributes:
    ----------
    epoch: {int-like}
    	当前的训练的epoch数目。
    verbose_round: {int-like}
    	每隔verbose_round轮次打印一次日志。
	loss: {list-like}
		保存每个epoch的Loss的数组。

    References:
    ----------
    [1] https://stackoverflow.com/questions/54888490/gensim-word2vec-print-log-loss
    '''
    def __init__(self, verbose_round=3):
        self.epoch = 0
        self.loss = []

        if verbose_round == 0:
            verbose_round = 1
        self.verbose_round = verbose_round

    def on_epoch_end(self, model):
        '''在每个epoch结束的时候计算模型的Loss并且打印'''

        # 获取该轮的Loss值
        loss = model.get_latest_training_loss()
        self.loss.append(loss)

        if len(self.loss) == 1:
            pass
        else:
            loss_decreasing_precent = np.round(
                (loss - self.loss[-2]) / self.loss[-2] * 100, 4
            )

            if divmod(self.epoch, self.verbose_round)[1] == 0:
                print('[INFO] {} Epoch {} word2vec loss: {:.2f}, decreasing {:.4f}%.'.format(
                    str(datetime.now())[:-7], self.epoch, loss, loss_decreasing_precent)
                )
        self.epoch += 1


class LoadSave():
    '''以*.pkl格式，利用pickle包存储各种形式（*.npz, list etc.）的数据。

    Attributes:
    ----------
        dir_name: {str-like}
            数据希望读取/存储的路径信息。
        file_name: {str-like}
            希望读取与存储的数据文件名。
        verbose: {int-like}
            是否打印存储路径信息。
    '''
    def __init__(self, dir_name=None, file_name=None, verbose=1):
        if dir_name is None:
            self.dir_name = './data_tmp/'
        else:
            self.dir_name = dir_name
        self.file_name = file_name
        self.verbose = verbose

    def save_data(self, dir_name=None, file_name=None, data_file=None):
        '''将data_file保存到dir_name下以file_name命名。'''
        if data_file is None:
            raise ValueError('LoadSave: Empty data_file !')

        if dir_name is None or not isinstance(dir_name, str):
            dir_name = self.dir_name
        if file_name is None:
            file_name = self.file_name
        if not isinstance(file_name, str) or not file_name.endswith('.pkl'):
            raise ValueError('LoadSave: Invalid file_name !')

        # 保存数据以指定名称到指定路径
        full_name = os.path.join(
            dir_name, file_name
        )
        with open(full_name, 'wb') as file_obj:
            pickle.dump(data_file, file_obj, protocol=4)

        if self.verbose:
            print('[INFO] {} LoadSave: Save to dir {} with name {}'.format(
                str(datetime.now())[:-7], dir_name, file_name)
            )

    def load_data(self, dir_name=None, file_name=None):
        '''从指定的dir_name载入名字为file_name的文件到内存里。'''
        if dir_name is None or not isinstance(dir_name, str):
            dir_name = self.dir_name
        if file_name is None:
            file_name = self.file_name
        if not isinstance(file_name, str) or not file_name.endswith('.pkl'):
            raise ValueError('LoadSave: Invalid file_name !')

        # 从指定路径导入指定文件名的数据
        full_name = os.path.join(
            dir_name, file_name
        )
        with open(full_name, 'rb') as file_obj:
            data_loaded = pickle.load(file_obj)

        if self.verbose:
            print('[INFO] {} LoadSave: Load from dir {} with name {}'.format(
                str(datetime.now())[:-7], dir_name, file_name)
            )
        return data_loaded
