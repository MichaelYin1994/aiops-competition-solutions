# -*- coding: utf-8 -*-

# Created on 202110090039
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
logging模块封装。
需求：
1. 能够在本地磁盘和命令行打印日志信息。
2. 日志信息包括：时间 - Logger的名称 - 级别 - 文件名：行号 - 日志关键信息。
3. 能够按照API发送日志到钉钉。
'''

import json
import logging
import urllib.request
from datetime import datetime

API_URL = 'https://oapi.dingtalk.com/robot/send?access_token=7f9e1d01046539141b1a1cbd378062153d3bbd9901a24b9c7f8824f125a51171'


def get_datetime():
    '''获取文件非法符号替换过的str时间戳'''
    curr_datetime = str(datetime.now()).split('.')[0]
    curr_date = curr_datetime.split(' ')[0]
    curr_time = curr_datetime.split(' ')[1]
    curr_time = curr_time.split(':')
    curr_time = '-'.join(curr_time)

    curr_datetime = '{} {}'.format(curr_date, curr_time)

    return curr_datetime


class DingTalkHandler(logging.Handler):
    '''用于给DingTalk发送日志的Handler，用法与其他Handler一致。'''
    def __init__(self):
        super(DingTalkHandler, self).__init__()
        self.info_queue = []

    def emit(self, log_record):
        info_text = self.format(log_record)

        # HTTP Head信息
        header = {
            'Content-Type': 'application/json',
            'Charset': 'UTF-8'
        }

        # 组装为json，并准备发送
        my_data = {
            'msgtype': 'markdown',
            'markdown': {'title': '[INFO] CURRENT FILE AT: {}'.format(datetime.now()),
                         'text': info_text},
            'at': {'isAtAll': False}
        }
        data_send = json.dumps(my_data)
        data_send = data_send.encode('utf-8')

        # 无网络连接，存入队列后续再发送
        def send_data(data_to_send):
            '''发送信息到指定API_URL'''
            request = urllib.request.Request(
                url=API_URL, data=data_to_send, headers=header
            )
            opener = urllib.request.urlopen(request)
            opener.read()

        try:
            # 队列不为空，有未发送的信息
            if self.info_queue:
                queue_length, n_pops = len(self.info_queue), 0

                for i in range(queue_length):
                    send_data(self.info_queue[i])
                    n_pops += 1

                for i in range(n_pops):
                    self.info_queue.pop(0)

            # 发送当前信息
            send_data(data_send)
        except:
            self.info_queue.append(data_send)


def get_logger(logger_name, logger_min_level='DEBUG', **kwargs):
    '''生成一个Logger的实例，用于日志的记录。'''
    if logger_name is None:
        logger_name = 'logger'

    is_print_std = kwargs.pop('is_print_std', True)
    is_send_dingtalk = kwargs.pop('is_send_dingtalk', False)
    is_save_to_disk = kwargs.pop('is_save_to_disk', False)
    log_path = kwargs.pop('log_path', None)

    if logger_min_level == 'DEBUG':
        logger_level = logging.DEBUG
    elif logger_min_level == 'WARNING':
        logger_level = logging.WARNING
    elif logger_min_level == 'INFO':
        logger_level = logging.INFO
    elif logger_min_level == 'CRITICAL':
        logger_level = logging.CRITICAL
    elif logger_min_level == 'ERROR':
        logger_level = logging.ERROR

    # 配置logger
    # ****************
    # Logger的配置项
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # 日志格式
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)8s | %(filename)15s:%(lineno)4d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 流处理器handler配置
    if is_print_std:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logger_level)
        stream_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)

    # 本地文件记录器的handler配置
    if is_save_to_disk and log_path and log_path.endswith('.log'):
        file_handler = logging.FileHandler(filename=log_path)
        file_handler.setLevel(logger_level)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    # Remote发送配置
    if is_send_dingtalk:
        dingtalk_handler = DingTalkHandler()
        dingtalk_handler.setLevel(logger_level)
        dingtalk_handler.setFormatter(formatter)

        logger.addHandler(dingtalk_handler)

    return logger
