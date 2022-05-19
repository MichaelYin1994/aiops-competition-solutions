# -*- coding: utf-8 -*-

# Created on 202203132027
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
基于TextCNN的分类模型。

References:
--------------
[1] https://github.com/blueloveTH/xwbank2020_baseline_keras/blob/master/models.py
[2] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
[3] He, Kaiming, et al. "Identity mappings in deep residual networks." European conference on computer vision. Springer, Cham, 2016.
[4] Zhang, Ye, and Byron Wallace. "A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification." arXiv preprint arXiv:1510.03820 (2015).

'''

import tensorflow as tf
import tensorflow.keras.backend as K

def loss(y_true, y_pred):

    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    weights = tf.constant([5 / 11, 4 / 11, 1 / 11, 1 / 11])
    weights = tf.cast(weights, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss


def resnet_block_conv1d(seq, n_filters, kernel_size):
    '''构建ResNet-like CONV-1D模块。'''
    # 维度转换(n_dim = n_filters * 4)

    # input: [n_batches, length, n_dim]
    # output: [n_batches, length, n_filters]
    x = tf.keras.layers.Conv1D(
        filters=n_filters, kernel_size=1,
        padding='same', activation='relu'
    )(seq)
    # x = tf.keras.layers.LayerNormalization()(x)

    # 特征抽取
    # input: [n_batches, length, n_filters]
    # output: [n_batches, length, n_filters]
    x = tf.keras.layers.Conv1D(
        filters=n_filters, kernel_size=kernel_size,
        padding='same', activation='relu'
    )(x)
    # x = tf.keras.layers.LayerNormalization()(x)

    # 维度提升
    # input: [n_batches, length, n_filters]
    # output: [n_batches, length, n_dim]
    x = tf.keras.layers.Conv1D(
        filters=int(n_filters * 4), kernel_size=kernel_size,
        padding='same', activation='relu'
    )(x)
    # x = tf.keras.layers.LayerNormalization()(x)

    # 残差连接
    # input: [n_batches, length, n_filters]
    # output: [n_batches, length, n_filters]
    x = tf.keras.layers.Add()([seq, x])
    # x = tf.keras.layers.concatenate([seq, x])

    return x


def block_cascade(x, kernel_size=5):
    '''将一系列ResNet-like Conv-1D模块进行级联'''
    # x = tf.keras.layers.Dropout(0.1)(x)

    # STEP 1: Block级联
    # --------
    # 第1层block
    x = tf.keras.layers.Conv1D(
        filters=128, kernel_size=1, padding='same', activation='relu'
    )(x)
    # x = tf.keras.layers.LayerNormalization()(x)
    x = resnet_block_conv1d(x, 32, kernel_size)
    # x = tf.keras.layers.SpatialDropout1D(0.1)(x)
    # x = tf.keras.layers.Dropout(0.1)(x)

    # # 第2层block
    # x = tf.keras.layers.Conv1D(
    #     filters=128, kernel_size=1, padding='same', activation='relu'
    # )(x)
    # # x = tf.keras.layers.LayerNormalization()(x)
    # x = resnet_block_conv1d(x, 32, kernel_size)
    # x = tf.keras.layers.Dropout(0.3)(x)

    # # 第3层block
    # x = tf.keras.layers.Conv1D(
    #     filters=64, kernel_size=1, padding='same', activation='relu'
    # )(x)
    # # x = tf.keras.layers.LayerNormalization()(x)
    # x = resnet_block_conv1d(x, 16, kernel_size)
    # x = tf.keras.layers.Dropout(0.3)(x)

    # STEP 2: 池化层
    # --------
    x_avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    x_max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)

    return [x_avg_pool, x_max_pool]


def build_model(verbose=False, is_compile=True, **kwargs):
    '''构建TextCNN分类模型'''

    # 初始化输入模型与参数
    # *******************
    model_configs = kwargs.pop('model_configs')
    n_classes = model_configs.pop('n_classes', 4)
    model_lr = model_configs.pop('model_lr', 0.00001)

    model_log_configs = model_configs['model_log_configs']
    model_template_configs = model_configs['model_template_configs']

    layer_input_token_ids = tf.keras.layers.Input(
        shape=(model_log_configs['max_len'], ), dtype=tf.int32
    )
    layer_input_time_seq = tf.keras.layers.Input(
        shape=(model_template_configs['max_len'], 2), dtype=tf.float32
    )
    layer_input_template_seq = tf.keras.layers.Input(
        shape=(model_template_configs['max_len'], ), dtype=tf.int32
    )
    layer_input_dense_feats = tf.keras.layers.Input(
        shape=(model_configs['n_dim_dense'], ), dtype=tf.float32
    )

    # 构建模型
    # *******************
    layer_dense_feats = tf.keras.layers.BatchNormalization()(layer_input_dense_feats)
    layer_dense_feats = tf.keras.layers.Dense(64, activation='tanh')(layer_dense_feats)
    layer_dense_feats = tf.keras.layers.BatchNormalization()(layer_input_dense_feats)
    # layer_dense_feats = tf.keras.layers.Dropout(0.1)(layer_dense_feats)

    layer_word_embedding = tf.keras.layers.Embedding(
        model_log_configs['max_vocab'] + 1,
        model_log_configs['embedding_size'],
        input_length=model_log_configs['max_len'],
        weights=[model_log_configs['embedding_mat']],
        name='layer_word_embedding',
        trainable=True
    )
    layer_template_embedding = tf.keras.layers.Embedding(
        model_template_configs['max_vocab'] + 1,
        model_template_configs['embedding_size'],
        input_length=model_template_configs['max_len'],
        weights=[model_template_configs['embedding_mat']],
        name='layer_template_embedding',
        trainable=True
    )

    layer_log_embedding = layer_word_embedding(layer_input_token_ids)
    layer_tid_embedding = layer_template_embedding(layer_input_template_seq)
    layer_tid_embedding = tf.keras.layers.concatenate(
        [layer_tid_embedding, layer_input_time_seq]
    )
    # layer_encoding_embedding = tf.keras.layers.Dropout(0.3)(layer_encoding_embedding)
    # layer_encoding_embedding = tf.keras.layers.SpatialDropout1D(0.3)(layer_encoding_embedding)
    layer_log_embedding = tf.keras.layers.LayerNormalization()(layer_log_embedding)
    layer_tid_embedding = tf.keras.layers.LayerNormalization()(layer_tid_embedding)

    layer_bilstm_log = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(layer_log_embedding)
    layer_bilstm_log = tf.keras.layers.LayerNormalization()(layer_bilstm_log)

    layer_bilstm_tid = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(layer_tid_embedding)
    layer_bilstm_tid = tf.keras.layers.LayerNormalization()(layer_bilstm_tid)

    # 组合，构建分类层
    # *******************
    layer_seq_feats = [
        tf.keras.layers.GlobalAveragePooling1D()(layer_bilstm_log),
        tf.keras.layers.GlobalMaxPooling1D()(layer_bilstm_log),
        tf.keras.layers.GlobalAveragePooling1D()(layer_bilstm_tid),
        tf.keras.layers.GlobalMaxPooling1D()(layer_bilstm_tid)
    ]

    layer_feat_concat = tf.keras.layers.concatenate(
        [
            # tf.keras.layers.GlobalAveragePooling1D()(layer_log_embedding),
            # tf.keras.layers.GlobalMaxPooling1D()(layer_log_embedding),            # tf.keras.layers.GlobalAveragePooling1D()(layer_encoding_embedding),
            # tf.keras.layers.GlobalAveragePooling1D()(layer_tid_embedding),            # tf.keras.layers.GlobalAveragePooling1D()(layer_encoding_embedding),
            # tf.keras.layers.GlobalMaxPooling1D()(layer_tid_embedding),
            layer_dense_feats
        ] + layer_seq_feats
    )

    layer_total_feat = tf.keras.layers.BatchNormalization()(layer_feat_concat)
    # layer_total_feat = tf.keras.layers.Dropout(0.1)(layer_total_feat)
    layer_total_feat = tf.keras.layers.Dense(64, activation='tanh')(layer_total_feat)

    # layer_total_feat = tf.keras.layers.BatchNormalization()(layer_total_feat)
    # layer_total_feat = tf.keras.layers.Dropout(0.1)(layer_total_feat)

    layer_output = tf.keras.layers.Dense(n_classes, activation='softmax')(layer_total_feat)

    # 编译模型
    # *******************
    model = tf.keras.models.Model(
        [layer_input_token_ids, layer_input_template_seq, layer_input_time_seq, layer_input_dense_feats],
        layer_output
    )
    model_feats = tf.keras.models.Model(
        [layer_input_token_ids, layer_input_template_seq, layer_input_time_seq, layer_input_dense_feats],
        layer_total_feat
    )

    if verbose:
        model.summary()

    if is_compile:
        model.compile(
            loss=[loss], optimizer=tf.keras.optimizers.Adam(model_lr)
        )

    return model, model_feats
