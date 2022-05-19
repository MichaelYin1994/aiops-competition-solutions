# -*- coding: utf-8 -*-

# Created on 202205150321
# Author:    zhuoyin94 <zhuoyin94@163.com>
# Github:    https://github.com/MichaelYin1994

'''
训练Lightgbm分类器，并采用最后lgb定轮次训练集成的方法。
'''

import argparse
import multiprocessing as mp
import os
import random

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, KFold

from utils.io_utils import LoadSave
from utils.logger import get_datetime, get_logger
from utils.metrics import compute_custom_score

GLOBAL_RANDOM_SEED = 9102

def custom_score_metric(y_true, y_pred_proba):
    y_pred_proba = y_pred_proba.reshape(len(np.unique(y_true)), -1)
    y_pred = np.argmax(y_pred_proba, axis=0)
    return 'custom_score', compute_custom_score(y_true, y_pred), True


def parse_opt():
    '''解析命令行参数'''
    parser = argparse.ArgumentParser()

    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--cv-strategy', type=str, default='gkf')
    parser.add_argument('--sub-style', type=str, default='offline')
    parser.add_argument('--early-stopping-rounds', type=int, default=1000)
    parser.add_argument('--n-estimators', type=int, default=7000)
    parser.add_argument('--keep-last-k-days', type=int, default=40)
    opt = parser.parse_args()

    return opt


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(GLOBAL_RANDOM_SEED)


if __name__ == '__main__':
    # 全局化的参数
    # *******************

    # Parse参数列表
    # ----------
    opt = parse_opt()

    # 全局化的参数
    # ----------
    N_FOLDS = opt.n_folds
    SUB_FIRST_K_FOLD = N_FOLDS
    CV_STRATEGY = opt.cv_strategy

    GLOBAL_OOV_TOKEN = -1
    EARLY_STOP_ROUNDS = opt.early_stopping_rounds
    MODEL_NAME = 'lgb_v2_fold_with_full_data'

    IS_SAVE_MODEL_TO_DISK = False
    IS_RANDOM_VISUALIZING = False
    IS_SEND_MSG_TO_DINGTALK = False
    IS_SAVE_LOG_TO_DISK = True

    # 配置日志格式
    # ----------
    LOGGING_PATH = './logs/'
    LOGGING_FILENAME = '{} {}.log'.format(
        get_datetime(), MODEL_NAME
    )

    logger = get_logger(
        logger_name=MODEL_NAME,
        is_print_std=True,
        is_send_dingtalk=IS_SEND_MSG_TO_DINGTALK,
        is_save_to_disk=IS_SAVE_LOG_TO_DISK,
        log_path=os.path.join(LOGGING_PATH, LOGGING_FILENAME)
    )

    # 打印相关参数
    # ----------
    logger.info(opt)

    # 特征工程数据载入
    # *******************
    file_handler = LoadSave(dir_name='./cached_data/lgb_feats/')
    train_feat_df, test_feat_df = file_handler.load_data(
        file_name='lgb_feat_df_w{}_list.pkl'.format(opt.keep_last_k_days)
    )

    logger.info(
        'Training lightgbm on: {}'.format('lgb_feat_df_w{}_list'.format(opt.keep_last_k_days))
    )

    # 模型训练与交叉验证
    # *******************

    # 训练数据元信息
    # ----------
    key_feat_names = ['sn', 'fault_time']
    category_feat_names = [
        name for name in train_feat_df.columns if 'category' in name
    ]
    numeric_feat_names = [
        name for name in train_feat_df.columns if 'numeric' in name
    ]

    train_feat_df = train_feat_df[train_feat_df['label'] != -1].reset_index(drop=True)
    train_target_vals = train_feat_df['label'].values
    train_feat_df.drop(['label'], axis=1, inplace=True)

    # 编码category变量
    # ----------
    for feat_name in category_feat_names:
        train_feat_df[feat_name] = train_feat_df[feat_name].astype('category')
        test_feat_df[feat_name] = test_feat_df[feat_name].astype('category')

    # for feat_name in train_feat_df.columns:
    #     if feat_name not in key_feat_names:
    #         # train_feat_df[feat_name] = train_feat_df[feat_name].fillna(0)
    #         # test_feat_df[feat_name] = test_feat_df[feat_name].fillna(0)

    #         train_feat_df[feat_name] = train_feat_df[feat_name].fillna(
    #             train_feat_df[feat_name].median()
    #         )
    #         test_feat_df[feat_name] = test_feat_df[feat_name].fillna(
    #             train_feat_df[feat_name].median()
    #         )

    # 交叉验证相关参数
    # ----------
    valid_score_cols = [
        'fold_id', 'valid_custom_score', 'valid_f1', 'valid_acc'
    ]
    valid_score_df = np.zeros((SUB_FIRST_K_FOLD, len(valid_score_cols)))
    test_pred_df_list = []
    oof_pred_proba_df = np.zeros((len(train_feat_df), 4))

    if CV_STRATEGY == 'kf':
        folds = KFold(n_splits=N_FOLDS)
        fold_generator = folds.split(
            np.arange(0, len(train_feat_df)),
        )
    elif CV_STRATEGY == 'gkf':
        folds = GroupKFold(n_splits=N_FOLDS)
        fold_generator = folds.split(
            np.arange(0, len(train_feat_df)), None,
            groups=train_feat_df['sn'].values
        )

    logger.info(
        'Cross validation strategy: {}'.format(CV_STRATEGY)
    )

    # 交叉验证
    # ----------
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'custom',
        'n_estimators': opt.n_estimators,
        'num_leaves': 31,
        'max_depth': 4,
        'learning_rate': 0.03,
        'colsample_bytree': 0.95,
        'subsample': 0.95,
        'subsample_freq': 1,
        'reg_alpha': 0,
        'reg_lambda': 0.001,
        'random_state': GLOBAL_RANDOM_SEED,
        'n_jobs': mp.cpu_count(),
        'verbose': -1
    }

    logger.info('TRAINING START...')
    logger.info('==================================')
    logger.info('train shape: {}'.format(train_feat_df.shape))
    for fold, (train_idx, valid_idx) in enumerate(fold_generator):

        # STEP 1: 依据训练与验证的bool索引, 构建训练与验证对应的数据集
        # ----------
        X_train = train_feat_df.iloc[train_idx].drop(key_feat_names, axis=1)
        X_valid = train_feat_df.iloc[valid_idx].drop(key_feat_names, axis=1)
        X_test = test_feat_df.drop(key_feat_names, axis=1)

        y_train = train_target_vals[train_idx]
        y_valid = train_target_vals[valid_idx]

        weights_mapping = {
            0: 5 / 11, 1: 4 / 11, 2: 1 / 11, 3: 1 / 11
        }
        sample_weights = np.array([weights_mapping[i] for i in y_train])

        logger.info(
            '-- train precent: {:.3f}%, valid precent: {:.3f}%'.format(
                100 * len(X_train) / len(train_feat_df),
                100 * len(X_valid) / len(train_feat_df)
            )
        )

        # STEP 2: 开始训练模型
        # ----------
        lgb_clf = lgb.LGBMClassifier(**lgb_params)
        lgb_clf.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_valid, y_valid)],
            eval_metric=custom_score_metric,
            early_stopping_rounds=EARLY_STOP_ROUNDS,
            verbose=False
        )

        # STEP 3: 预测与评估
        # ----------
        y_val_pred = lgb_clf.predict_proba(
            X_valid, num_iteration=lgb_clf.best_iteration_ + 1
        )
        y_val_pred_label = np.argmax(y_val_pred, axis=1)

        # 存储测试预测DataFrame
        sub_tmp_df = test_feat_df.copy()[key_feat_names]
        test_pred_proba_tmp = lgb_clf.predict_proba(
            X_test, num_iteration=lgb_clf.best_iteration_ + 1
        )
        test_pred_proba_tmp_df = pd.DataFrame(
            test_pred_proba_tmp, columns=['class_proba_{}'.format(i) for i in range(4)]
        )
        test_pred_df_list.append(
            pd.concat([sub_tmp_df, test_pred_proba_tmp_df], axis=1)
        )

        # 存储valid的oof预测结果
        oof_pred_proba_df[valid_idx, :] = y_val_pred

        val_custom = compute_custom_score(
            y_valid,
            y_val_pred_label
        )
        val_f1 = f1_score(
            y_valid.reshape(-1, 1),
            y_val_pred_label.reshape(-1, 1),
            average='macro'
        )
        val_acc = accuracy_score(
            y_valid.reshape(-1, 1),
            y_val_pred_label.reshape(-1, 1)
        )

        logger.info(
            '-- fold {}({}): val custom: {:.4f}, f1: {:.4f}, acc: {:.4f}, best iters: {}\n'.format(
                fold+1, SUB_FIRST_K_FOLD, val_custom, val_f1, val_acc, lgb_clf.best_iteration_ + 1
            )
        )

        valid_score_df[fold, 0] = fold
        valid_score_df[fold, 1] = val_custom
        valid_score_df[fold, 2] = val_f1
        valid_score_df[fold, 3] = val_acc

    # 整体Out of fold训练指标估计
    # *******************
    valid_score_df = pd.DataFrame(
        valid_score_df, columns=valid_score_cols
    )
    valid_score_df['fold_id'] = valid_score_df['fold_id'].astype(int)

    train_oof_pred_vals = np.argmax(oof_pred_proba_df, axis=-1)
    global_custom = compute_custom_score(
        train_target_vals,
        train_oof_pred_vals
    )
    global_f1 = f1_score(
        train_target_vals.reshape(-1, 1),
        train_oof_pred_vals.reshape(-1, 1),
        average='macro'
    )
    global_acc = accuracy_score(
        train_target_vals.reshape(-1, 1),
        train_oof_pred_vals.reshape(-1, 1)
    )

    logger.info(
        '-- TOTAL OOF: val custom: {:.4f}, f1: {:.4f}, acc: {:.4f}'.format(
            global_custom, global_f1, global_acc
        )
    )
    logger.info(
        '\n' + str(classification_report(
            train_target_vals, train_oof_pred_vals, digits=4)
        )
    )
    logger.info(
        '-- TOTAL AVG: val custom: {:.4f}, f1: {:.4f}, acc: {:.4f}'.format(
            valid_score_df['valid_custom_score'].mean(),
            valid_score_df['valid_f1'].mean(),
            valid_score_df['valid_acc'].mean()
        )
    )
    logger.info('\n' + str(valid_score_df))

    logger.info('TRAINING END...')
    logger.info('==================================')

    # 全样本定轮次训练
    # *******************
    logger.info('A.T. FIELD, FULL POWER ! ! !')

    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'custom',
        'n_estimators': 1000,
        'num_leaves': 31,
        'max_depth': 4,
        'learning_rate': 0.03,
        'colsample_bytree': 0.95,
        'subsample': 0.95,
        'subsample_freq': 1,
        'reg_alpha': 0,
        'reg_lambda': 0.001,
        'random_state': GLOBAL_RANDOM_SEED,
        'n_jobs': mp.cpu_count(),
        'verbose': -1
    }

    # STEP 1: 构建训练与验证对应的数据集
    # ----------
    X_train = train_feat_df.drop(key_feat_names, axis=1)
    X_test = test_feat_df.drop(key_feat_names, axis=1)

    y_train = train_target_vals
    weights_mapping = {
        0: 5 / 11, 1: 4 / 11, 2: 1 / 11, 3: 1 / 11
    }
    sample_weights = np.array([weights_mapping[i] for i in y_train])

    # STEP 2: 开始训练模型
    # ----------
    logger.info('FULL POWER shape: {}'.format(X_train.shape))

    lgb_clf = lgb.LGBMClassifier(**lgb_params)
    lgb_clf.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_metric=custom_score_metric,
        verbose=False
    )

    # STEP 3: 获取预测结果
    # ----------
    sub_tmp_df = test_feat_df.copy()[key_feat_names]
    test_pred_proba_tmp = lgb_clf.predict_proba(
        X_test, num_iteration=lgb_params['n_estimators'] - 1
    )
    test_pred_proba_tmp_df = pd.DataFrame(
        test_pred_proba_tmp, columns=['class_proba_{}'.format(i) for i in range(4)]
    )
    test_pred_df_list.append(
        pd.concat([sub_tmp_df, test_pred_proba_tmp_df], axis=1)
    )

    # 预测结果集成
    # *******************
    file_processor = LoadSave(dir_name='./cached_data')
    test_pred_df_list.append(
        file_handler.load_data(file_name='xiaosheng_pred.pkl')
    )

    # 预测与cv结果保存
    # *******************
    if 'submissions' not in os.listdir():
        os.mkdir('./submissions/')
    if 'submissions_oof' not in os.listdir():
        os.mkdir('./submissions_oof/')

    FILE_NAME = '{}_{}_nfold_{}_valcustom_{}_f1_{}'.format(
        len(os.listdir('./submissions')) + 1,
        MODEL_NAME, N_FOLDS,
        str(np.round(global_custom, 4)).split('.')[1],
        str(np.round(global_f1, 4)).split('.')[1],
    )

    sub_df = test_feat_df.copy()[key_feat_names]
    test_pred_list = [
        df.drop(key_feat_names, axis=1).values for df in test_pred_df_list
    ]
    test_pred_label = np.argmax(np.average(
        test_pred_list, axis=0,
        weights=[0.08, 0.08, 0.08, 0.08, 0.08, 0.5, 0.1]
    ), axis=1)
    # test_pred_label = np.argmax(np.mean(test_pred_list, axis=0), axis=1)
    sub_df['label'] = test_pred_label

    # 保存预测结果
    sub_df.to_csv(
        './submissions/{}.csv'.format(FILE_NAME), index=False
    )
    logger.info('Save succesed ! Submission file name: {}.csv'.format(FILE_NAME))

    sub_df.to_csv('./result.csv', index=False)
    logger.info('Save succesed ! Submission file name: result.csv')

    # # 保存cv结果
    # oof_pred_proba_df = pd.DataFrame(
    #     oof_pred_proba_df, columns=['class_proba_{}'.format(i) for i in range(4)]
    # )
    # oof_pred_proba_df['sn'] = train_feat_df['sn']
    # oof_pred_proba_df['label'] = train_target_vals

    # file_handler = LoadSave(dir_name='./submissions_oof/')
    # file_handler.save_data(
    #     file_name='{}.pkl'.format(FILE_NAME),
    #     data_file=[oof_pred_proba_df, test_pred_df_list]
    # )

    logger.info('\n***************FINISHED...***************')
