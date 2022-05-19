# -*- coding: utf-8 -*-

# Created on 202203132027
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
基于BERT-like的分类模型。
'''

import tensorflow as tf


def build_model(verbose=False, is_compile=True, **kwargs):
    '''构建BERT-based的长文本NER模型'''
    # 初始化输入模型与参数
    # *******************
    model_contextual = kwargs.pop('model_contextual')
    model_contextual_configs = kwargs.pop('model_contextual_configs')
    n_classes = kwargs.pop('n_classes', 4)

    layer_input_token_ids = tf.keras.layers.Input(
        shape=(model_contextual_configs['max_len'], ), dtype=tf.int32
    )
    layer_input_attention_mask = tf.keras.layers.Input(
        shape=(model_contextual_configs['max_len'], ), dtype=tf.int32
    )
    layer_input_dense_feats = tf.keras.layers.Input(
        shape=(model_contextual_configs['n_dim_dense'], ), dtype=tf.float32
    )

    # model_contextual.trainable = False
    layer_contextual_output = model_contextual(
        input_ids=layer_input_token_ids,
        attention_mask=layer_input_attention_mask
    )[0]

    # 模型构造
    # *******************
    layer_cls_vec = tf.keras.layers.Dropout(0.3)(
        tf.keras.layers.Lambda(lambda x: x[:, 0])(layer_contextual_output)
    )

    # layer_avg_vec = tf.keras.layers.Dropout(0.3)(
    #     tf.keras.layers.GlobalAveragePooling1D()(layer_contextual_output)
    # )

    x = tf.keras.layers.Dense(256, activation='relu')(layer_cls_vec)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    y = tf.keras.layers.Dense(64, activation='relu')(layer_input_dense_feats)
    y = tf.keras.layers.Dropout(0.2)(y)

    x = tf.keras.layers.concatenate([x, y])
    x = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    model = tf.keras.models.Model(
        [layer_input_token_ids, layer_input_attention_mask, layer_input_dense_feats], x
    )

    # 模型训练参数
    # *******************
    if verbose:
        model.summary()

    if is_compile:
        model.compile(
            loss=['categorical_crossentropy'],
            optimizer=tf.keras.optimizers.Adam(model_contextual_configs['model_lr']),
            metrics=[tf.keras.metrics.AUC(num_thresholds=1500), 'acc']
        )

    return model

