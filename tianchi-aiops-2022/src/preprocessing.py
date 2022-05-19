# -*- coding: utf-8 -*-

# Created on 202202260047
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
对原始数据的预处理工作。
'''

import os

import pandas as pd
from utils.io_utils import LoadSave

TRAIN_DATA_DIR = './data'
TEST_DATA_DIR = './tcdata'

def split_venus_log(meta_df):
    '''切分venus_log_df细节信息'''
    module_list = meta_df['module'].split(',')
    module_cause_list = meta_df['module_cause'].split(',')

    split_corpus_list = []
    for module_name in module_list:

        # 找到module cause details开始的idx
        for start_idx in range(len(module_cause_list)):
            if module_cause_list[start_idx] == module_name:
                break

        # 找到module cause details结束的idx
        end_idx = start_idx + 1
        while(end_idx <= len(module_cause_list)):
            if end_idx != len(module_cause_list) and module_cause_list[end_idx] not in module_list:
                end_idx += 1
            else:
                break

        # 保存结果
        split_corpus_list.append(
            [
                meta_df['sn'],
                meta_df['fault_time'],
                module_name,
                module_cause_list[start_idx+1:end_idx]
            ]
        )
    return split_corpus_list


def split_crashdump_log(fault_code_info):
    '''切分crashdump_log_df细节信息'''
    split_info = fault_code_info.split('.')

    split_info[-1] = int(split_info[-1], base=16)
    return split_info


def processing_venus_df(venus_df):
    '''对venus_df表进行预处理'''
    # 拆分venus_log_df数据
    venus_log_list_tmp = venus_df.apply(split_venus_log, axis=1)
    venus_log_list = []
    for item in venus_log_list_tmp:
        venus_log_list.extend(item)

    # 构建新的venus_df
    venus_df = pd.DataFrame(
        venus_log_list, columns=['sn', 'fault_time', 'module', 'module_cause']
    )

    venus_df['module_cause_cod0'] = venus_df['module_cause'].apply(
        lambda x: int(x[0].split(':')[1], base=16) if len(x) == 3 else -1
    )
    venus_df['module_cause_cod1'] = venus_df['module_cause'].apply(
        lambda x: int(x[1].split(':')[1], base=16) if len(x) == 3 else -1
    )
    venus_df['module_cause_addr'] = venus_df['module_cause'].apply(
        lambda x: int(x[2].split(':')[1], base=16) if len(x) == 3 else -1
    )
    venus_df['module_cause_others'] = venus_df['module_cause'].apply(
        lambda x: x[0] if len(x) == 1 else None
    )
    venus_df.drop(['module_cause'], axis=1, inplace=True)

    return venus_df


def processing_crashdump_df(crashdump_df):
    '''对crashdump_df进行预处理'''
    crashdump_info_list = crashdump_df['fault_code'].apply(split_crashdump_log)

    crashdump_df['fault_code_0'] = [item[0] for item in crashdump_info_list]
    crashdump_df['fault_code_1'] = [item[1] for item in crashdump_info_list]
    crashdump_df['fault_code_2'] = [item[2] for item in crashdump_info_list]
    crashdump_df['fault_code_3'] = [item[3] for item in crashdump_info_list]

    crashdump_df.drop(['fault_code'], axis=1, inplace=True)

    return crashdump_df


if __name__ == '__main__':
    # 导入所有数据
    # *******************
    sel_log_list, label_list = [], []

    # 初赛A轮train数据
    train_sel_log_df = pd.read_csv(
        os.path.join(TRAIN_DATA_DIR, 'preliminary_sel_log_dataset.csv')
    )
    train_label_a_df = pd.read_csv(
        os.path.join(TRAIN_DATA_DIR, 'preliminary_train_label_dataset.csv')
    )
    train_label_b_df = pd.read_csv(
        os.path.join(TRAIN_DATA_DIR, 'preliminary_train_label_dataset_s.csv')
    )

    train_venus_df = pd.read_csv(
        os.path.join(TRAIN_DATA_DIR, 'preliminary_venus_dataset.csv')
    )
    train_venus_df = processing_venus_df(train_venus_df)

    train_crashdump_df = pd.read_csv(
        os.path.join(TRAIN_DATA_DIR, 'preliminary_crashdump_dataset.csv')
    )
    train_crashdump_df = processing_crashdump_df(train_crashdump_df)

    sel_log_list.append(train_sel_log_df)
    label_list.extend([train_label_a_df, train_label_b_df])

    # 初赛语料数据
    additional_sel_log_df = pd.read_csv(
        os.path.join(TRAIN_DATA_DIR, 'additional_sel_log_dataset.csv')
    )

    # 初赛A轮test数据
    test_sel_log_a_df = pd.read_csv(
        os.path.join(TRAIN_DATA_DIR, 'preliminary_sel_log_dataset_a.csv')
    )
    test_label_a_df = pd.read_csv(
        os.path.join(TRAIN_DATA_DIR, 'preliminary_submit_dataset_a.csv')
    )
    test_label_a_df['label'] = -1

    sel_log_list.append(test_sel_log_a_df)
    label_list.append(test_label_a_df)

    # 初赛B轮test数据
    test_sel_log_b_df = pd.read_csv(
        os.path.join(TRAIN_DATA_DIR, 'preliminary_sel_log_dataset_b.csv')
    )
    test_label_b_df = pd.read_csv(
        os.path.join(TRAIN_DATA_DIR, 'preliminary_submit_dataset_b.csv')
    )
    test_label_b_df['label'] = -1

    sel_log_list.append(test_sel_log_b_df)
    label_list.append(test_label_b_df)

    # 复赛B轮数据
    test_sel_log_c_df = pd.read_csv(
        os.path.join(TEST_DATA_DIR, 'final_sel_log_dataset_b.csv')
    )
    test_label_c_df = pd.read_csv(
        os.path.join(TEST_DATA_DIR, 'final_submit_dataset_b.csv')
    )
    test_venus_c_df = pd.read_csv(
        os.path.join(TEST_DATA_DIR, 'final_venus_dataset_b.csv')
    )
    test_crashdump_c_df = pd.read_csv(
        os.path.join(TEST_DATA_DIR, 'final_crashdump_dataset_b.csv')
    )

    test_venus_c_df = processing_venus_df(test_venus_c_df)
    test_crashdump_c_df = processing_crashdump_df(test_crashdump_c_df)

    # 数据预处理
    # *******************
    train_sel_log_df = pd.concat(sel_log_list, axis=0).reset_index(drop=True)
    train_label_df = pd.concat(label_list, axis=0).reset_index(drop=True)

    train_df_list = [
        train_sel_log_df,
        train_label_df,
        additional_sel_log_df,
        train_venus_df,
        train_crashdump_df,
    ]
    test_df_list = [
        test_sel_log_c_df,
        test_label_c_df,
        test_venus_c_df,
        test_crashdump_c_df
    ]

    # 转换日期为Datetime的对象
    for df in train_df_list + test_df_list:
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.sort_values(
                by=['sn', 'time'], ascending=[True, True], inplace=True
            )

        if 'fault_time' in df.columns:
            df['fault_time'] = pd.to_datetime(df['fault_time'])
            df.sort_values(
                by=['sn', 'fault_time'], ascending=[True, True], inplace=True
            )

        df.reset_index(drop=True, inplace=True)

    # 预处理数据存储
    # *******************
    if 'cached_data' not in os.listdir():
        os.mkdir('./cached_data')

    file_handler = LoadSave(dir_name='./cached_data/')
    file_handler.save_data(
        file_name='train_df_list.pkl', data_file=train_df_list
    )
    file_handler.save_data(
        file_name='test_df_list.pkl', data_file=test_df_list
    )
