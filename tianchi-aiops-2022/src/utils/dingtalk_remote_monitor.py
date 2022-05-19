# -*- coding: utf-8 -*-

# Created on 202009082036
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
利用钉钉（DingTalk）的API构建的训练监控机器人。
'''

import tensorflow.keras as keras
import numpy as np


class RemoteMonitorDingTalk(keras.callbacks.Callback):
    '''
    在神经网络每一个batch训练结束之后，发送train和validation的信息给远程钉钉服务器。

    Attributes:
    ----------
    model_name: {str-like}
        该次训练的模型的名字。
    gpu_id: {int-like}
        当前模型使用的GPU的ID编号。
    logger: {object-like}
        logging模块的实例，用于存储日志到本地。
    '''
    def __init__(self, model_name=None, gpu_id=0, logger=None):
        super(RemoteMonitorDingTalk, self).__init__()
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.logger = logger

    def on_epoch_end(self, epoch, logs):
        '''在每一个epoch之后，发送logs信息到远程服务器。'''
        log_keys = list(logs.keys())
        for k in log_keys:
            logs[k] = np.round(logs[k], 5)

        info_text = str(logs)
        if self.model_name is None:
            self.logger.info(
                'EPOCH: {}, '.format(epoch) + info_text
            )
        else:
            self.logger.info(
                '[{}][GPU: {}] EPOCH: {}, '.format(self.model_name, self.gpu_id, epoch) + info_text
            )
