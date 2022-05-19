#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202102041520
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

"""
数据处理与特征工程辅助代码。
"""

import pickle
import warnings
from datetime import datetime

import numba
import numpy as np
import pandas as pd
import tensorflow as tf
from numba import njit
from tqdm import tqdm

warnings.filterwarnings("ignore")
###############################################################################

# 全局变量：观测窗口时间
INTERVAL_SECONDS = int(np.ceil(5 * 60))

class LoadSave():
    """以*.pkl格式，利用pickle包存储各种形式（*.npz, list etc.）的数据。

    @Attributes:
    ----------
        dir_name: {str-like}
            数据希望读取/存储的路径信息。
        file_name: {str-like}
            希望读取与存储的数据文件名。
        verbose: {int-like}
            是否打印存储路径信息。
    """
    def __init__(self, dir_name=None, file_name=None, verbose=1):
        if dir_name is None:
            self.dir_name = "./data_tmp/"
        else:
            self.dir_name = dir_name
        self.file_name = file_name
        self.verbose = verbose

    def save_data(self, dir_name=None, file_name=None, data_file=None):
        """将data_file保存到dir_name下以file_name命名。"""
        if data_file is None:
            raise ValueError("LoadSave: Empty data_file !")

        if dir_name is None or not isinstance(dir_name, str):
            dir_name = self.dir_name
        if file_name is None:
            file_name = self.file_name
        if not isinstance(file_name, str) or not file_name.endswith(".pkl"):
            raise ValueError("LoadSave: Invalid file_name !")

        # 保存数据以指定名称到指定路径
        full_name = dir_name + file_name
        with open(full_name, "wb") as file_obj:
            pickle.dump(data_file, file_obj, protocol=4)

        if self.verbose:
            print("[INFO] {} LoadSave: Save to dir {} with name {}".format(
                str(datetime.now())[:-4], dir_name, file_name))

    def load_data(self, dir_name=None, file_name=None):
        """从指定的dir_name载入名字为file_name的文件到内存里。"""
        if dir_name is None or not isinstance(dir_name, str):
            dir_name = self.dir_name
        if file_name is None:
            file_name = self.file_name
        if not isinstance(file_name, str) or not file_name.endswith(".pkl"):
            raise ValueError("LoadSave: Invalid file_name !")

        # 从指定路径导入指定文件名的数据
        full_name = dir_name + file_name
        with open(full_name, "rb") as file_obj:
            data_loaded = pickle.load(file_obj)

        if self.verbose:
            print("[INFO] {} LoadSave: Load from dir {} with name {}".format(
                str(datetime.now())[:-4], dir_name, file_name))
        return data_loaded


class CategoricalEncoder():
    """对于输入的array对象的元素进行重新encoding，赋予新的编号。

    扫描数组内容，构建category2id的字典，通过字典替换的方法进行编码。

    @Attributes:
    ----------
    is_encoding_nan: {bool-like}
        是否对能够被np.isnan方法检测的对象进行编码，默认为不需要编码。
    cat2id: {dict-like}
        原始的category对象到新编码id的映射表。
    id2cat: {dict-like}
        新编码id到原始的category对象的映射表。

    """
    def __init__ (self, is_encoding_nan=False):
        # TODO(zhuoyin94@163.com): 后续添加对nan处理的支持
        self.is_encoding_nan = is_encoding_nan

    def fit(self, raw_data=None):
        """扫描raw_data，利用字典记录raw_data的unique的id对象，返回None值。"""
        if isinstance(raw_data, (list, np.ndarray)):
            pass
        else:
            raise TypeError("Invalid input data type !")

        # 类别到id与id到类别的映射
        self.cat2id, self.id2cat = {}, {}

        global_id = 0
        for item in raw_data:
            # FIXME(zhuoyin94@163.com): 不安全的检测方法
            if item not in self.cat2id and item is not np.nan:
                self.cat2id[item] = global_id
                self.id2cat[global_id] = item
                global_id += 1

        return None

    def fit_transform(self, raw_data=None, inplace=True):
        """扫描raw_data，利用字典记录raw_data的unique的id对象，返回转换后的数组。"""
        if isinstance(raw_data, (list, np.ndarray)):
            pass
        else:
            raise TypeError("Invalid input data type !")

        if inplace is False:
            transformed_data = np.zeros(len(raw_data), )

        # 类别到id与id到类别的映射
        self.cat2id, self.id2cat = {}, {}

        global_id = 0
        for idx, item in enumerate(raw_data):
            # FIXME(zhuoyin94@163.com): 不安全的检测方法
            if item not in self.cat2id and item is not np.nan:
                self.cat2id[item] = global_id
                self.id2cat[global_id] = item
                global_id += 1

            if item is not np.nan and inplace is True:
                raw_data[idx] = global_id
            elif item is not np.nan and inplace is False:
                transformed_data[idx] = global_id

        # Returen results
        if inplace:
            return raw_data
        else:
            return transformed_data

    def transform(self, raw_data=None):
        """转换新输入的对象，并返回转换后的对象。"""
        if len(self.cat2id) == 0 or len(self.id2cat) == 0:
            raise ValueError("Please fit first !")
        if len(raw_data) == 0:
            return np.array([])

        for idx in range(len(raw_data)):
            if raw_data[idx] in self.cat2id:
                raw_data[idx] = self.cat2id[raw_data[idx]]
        return raw_data

    def reverse_transform(self, transformed_data=None):
        """将被转换的数据恢复为原有的编码。"""
        if len(self.cat2id) == 0 or len(self.id2cat) == 0:
            raise ValueError("Please fit first !")
        if len(transformed_data) == 0:
            return np.array([])

        for idx in range(len(transformed_data)):
            if transformed_data[idx] in self.id2cat:
                transformed_data[idx] = self.id2cat[transformed_data[idx]]
        return transformed_data


def create_time2bin_numba_dict():
    """创建高效的将unix时间戳映射到制定间隔的bin的numba字典，并返回该numba字典。"""
    # 1546300800
    start_date = pd.to_datetime(["2019-01-01 00:00:00"]).astype(int)
    start_date = int(start_date[0] / 10**9)

    # 1568505600
    end_date = pd.to_datetime(["2019-10-02 00:00:00"]).astype(int)
    end_date = int(end_date[0] / 10**9)

    dt_bins = np.arange(
        start_date, end_date+INTERVAL_SECONDS, INTERVAL_SECONDS)
    dt_bins[0] = int(dt_bins[0] - 10)
    dt_labels = [i for i in range(len(dt_bins) - 1)]

    # 构造时序字典DataFrame
    time2label_df = pd.DataFrame(None)

    time2label_df["unix_time"] = np.arange(
        start_date - 10, end_date)
    time2label_df["unix_time_bin"] = pd.cut(
        time2label_df["unix_time"].values,
        bins=dt_bins, labels=dt_labels).codes
    time2label_df["unix_time_bin"] = \
        time2label_df["unix_time_bin"].values.astype(np.int64)

    time2label_df["unix_time_dt"] = \
        pd.to_datetime(time2label_df["unix_time"].values, unit="s")
    time2label_df["global_day"] = time2label_df["unix_time_dt"] \
            - pd.to_datetime("2019-01-01 00:00:00")
    time2label_df["global_day"] = time2label_df["global_day"].dt.days
    time2label_df["global_day"] = \
        time2label_df["global_day"].astype(np.int64)

    time2label_df["global_hour"] = time2label_df["unix_time_dt"] \
                - pd.to_datetime("2019-01-01 00:00:00")
    time2label_df["global_hour"] = \
        time2label_df["global_hour"].dt.total_seconds() / 3600
    time2label_df["global_hour"] = np.floor(time2label_df["global_hour"].values)
    time2label_df["global_hour"] = \
        time2label_df["global_hour"].astype(np.int64)

    time2label_df["collect_time_bin_right_edge"] = \
        dt_bins[1:][time2label_df["unix_time_bin"].values]
    time2label_df["collect_time_bin_right_edge"] = \
        time2label_df["collect_time_bin_right_edge"].astype(np.int64)

    time2label_df.drop(["unix_time_dt"], axis=1, inplace=True)

    # 构造时序索引字典
    time2label_dict = numba.typed.Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.int64[:])
    time2label_vals = time2label_df.values

    for idx in tqdm(range(len(time2label_vals))):
        time2label_dict[int(time2label_vals[idx][0])] = \
            time2label_vals[idx][1:]
    return time2label_dict


@njit
def njit_compute_time2bin(unix_time_array, time2bin_dict, n_feats):
    "将unix_time_array中的unix时间映射到对应的采样区间上去，并使用njit进行加速。"
    time2bin_res = np.zeros((len(unix_time_array), n_feats))

    for i in range(len(unix_time_array)):
        time2bin_res[i, :] = time2bin_dict[unix_time_array[i]]
    return time2bin_res


@njit
def njit_compute_vendor2vec(vendor_array, vendor2vec):
    "将vendor_array中的vendor信息映射到对应的vector表示上，并使用njit进行加速。"
    vendor_feat = np.zeros((len(vendor_array), 3))
    for i, vendor in enumerate(vendor_array):
        if vendor == 0:
            vendor_feat[i, :] = vendor2vec[0, :]
        elif vendor == 1:
            vendor_feat[i, :] = vendor2vec[1, :]
        elif vendor == 2:
            vendor_feat[i, :] = vendor2vec[2, :]
        else:
            vendor_feat[i, :] = vendor2vec[0, :]
    return vendor_feat


@njit
def njit_compute_manufacturer2vec(
        manufacturer_array, manufacturer2vec):
    "将manufacturer_array中的manufacturer信息映射到对应的vector表示上，并使用njit进行加速。"
    manufacturer_feat = np.zeros(
        (len(manufacturer_array), 4))
    for i, manufacturer in enumerate(manufacturer_array):
        if manufacturer == 0:
            manufacturer_feat[i, :] = manufacturer2vec[0, :]
        elif manufacturer == 1:
            manufacturer_feat[i, :] = manufacturer2vec[1, :]
        elif manufacturer == 2:
            manufacturer_feat[i, :] = manufacturer2vec[2, :]
        elif manufacturer == 3:
            manufacturer_feat[i, :] = manufacturer2vec[3, :]
        else:
            manufacturer_feat[i, :] = manufacturer2vec[0, :]
    return manufacturer_feat


@njit
def njit_compute_transaction2vec(transaction_array, transaction2vec):
    "将transaction_array中的vendor信息映射到对应的vector表示上，并使用njit进行加速。"
    transaction_feat = np.zeros((len(transaction_array), 4))
    for i, transaction in enumerate(transaction_array):
        if transaction == 0:
            transaction_feat[i, :] = transaction2vec[0, :]
        elif transaction == 1:
            transaction_feat[i, :] = transaction2vec[1, :]
        elif transaction == 2:
            transaction_feat[i, :] = transaction2vec[2, :]
        elif transaction == 3:
            transaction_feat[i, :] = transaction2vec[3, :]
        else:
            transaction_feat[i, :] = transaction2vec[0, :]
    return transaction_feat


def load_csv(dir_name, file_name, nrows=100, **kwargs):
    """从指定路径dir_name读取名为file_name的*.csv文件，nrows指定读取前nrows行。"""
    if dir_name is None or file_name is None or not file_name.endswith(".csv"):
        raise ValueError("Invalid dir_name or file_name !")

    full_name = dir_name + file_name
    data = pd.read_csv(full_name, nrows=nrows, **kwargs)
    return data


def basic_feature_report(data_table, quantile=None):
    """抽取Pandas的DataFrame的基础信息。"""
    if quantile is None:
        quantile = [0.25, 0.5, 0.75, 0.95, 0.99]

    # 基础统计数据
    data_table_report = data_table.isnull().sum()
    data_table_report = pd.DataFrame(data_table_report, columns=["#missing"])

    data_table_report["#uniques"] = data_table.nunique(dropna=False).values
    data_table_report["types"] = data_table.dtypes.values
    data_table_report.reset_index(inplace=True)
    data_table_report.rename(columns={"index": "feature_name"}, inplace=True)

    # 分位数统计特征
    data_table_description = data_table.describe(quantile).transpose()
    data_table_description.reset_index(inplace=True)
    data_table_description.rename(
        columns={"index": "feature_name"}, inplace=True)
    data_table_report = pd.merge(
        data_table_report, data_table_description,
        on="feature_name", how="left")

    return data_table_report


def save_as_csv(df, dir_name, file_name):
    """将Pandas的DataFrame以*.csv的格式保存到dir_name+file_name路径。"""
    if dir_name is None or file_name is None or not file_name.endswith(".csv"):
        raise ValueError("Invalid dir_name or file_name !")

    full_name = dir_name + file_name
    df.to_csv(full_name, index=False)


class ReduceMemoryUsage():
    """通过pandas的column的强制类型转换降低pandas数据表的内存消耗。
    返回经过类型转换后的DataFrame。

    @Attributes:
    ----------
    data_table: pandas DataFrame-like
        需要降低内存消耗的pandas的DataFrame对象。
    verbose: bool
        是否打印类型转换前与转换后的相关信息。

    @Return:
    ----------
    类型转换后的pandas的DataFrame数组。

    @References:
    ----------
    [1] https://docs.scipy.org/doc/numpy/reference/generated/numpy.iinfo.html
    [2] https://wizard
    forcel.gitbooks.io/ts-numpy-tut/content/3.html
    """
    def __init__(self, data_table=None, verbose=True):
        self.data_table = data_table
        self.verbose = verbose

    def get_dataframe_types(self, data_table):
        """获取pandas的DataFrame的每一列的数据类型，并返回类型的字典"""
        data_table_types = list(map(str, data_table.dtypes.values))

        type_dict = {}
        for ind, name in enumerate(data_table.columns):
            type_dict[name] = data_table_types[ind]
        return type_dict

    def reduce_memory_usage(self):
        """对self.data_table的每一列进行类型转换，返回经过转换后的DataFrame。"""
        memory_usage_before_transformed = self.data_table.memory_usage(
            deep=True).sum() / 1024**2
        type_dict = self.get_dataframe_types(self.data_table)

        if self.verbose is True:
            print("[INFO] {} Reduce memory usage:".format(
                str(datetime.now())[:-4]))
            print("----------------------------------")
            print("[INFO] {} Memory usage of data is {:.5f} MB.".format(
                str(datetime.now())[:-4], memory_usage_before_transformed))

        # 扫描每一个column，若是属于float或者int类型，则进行类型转换
        for name in tqdm(list(type_dict.keys())):
            feat_type = type_dict[name]

            if "float" in feat_type or "int" in feat_type:
                feat_min = self.data_table[name].min()
                feat_max = self.data_table[name].max()

                if "int" in feat_type:
                    if feat_min > np.iinfo(np.int8).min and \
                        feat_max < np.iinfo(np.int8).max:
                        self.data_table[name] = \
                            self.data_table[name].astype(np.int8)
                    elif feat_min > np.iinfo(np.int16).min and \
                        feat_max < np.iinfo(np.int16).max:
                        self.data_table[name] = \
                            self.data_table[name].astype(np.int16)
                    elif feat_min > np.iinfo(np.int32).min and \
                        feat_max < np.iinfo(np.int32).max:
                        self.data_table[name] = \
                            self.data_table[name].astype(np.int32)
                    else:
                        self.data_table[name] = \
                            self.data_table[name].astype(np.int64)
                else:
                    if feat_min > np.finfo(np.float32).min and \
                        feat_max < np.finfo(np.float32).max:
                        self.data_table[name] = \
                            self.data_table[name].astype(np.float32)
                    else:
                        self.data_table[name] = \
                            self.data_table[name].astype(np.float64)

        memory_usage_after_reduced = self.data_table.memory_usage(
            deep=True).sum() / 1024**2
        if self.verbose is True:
            print("\n----------------------------------")
            print("[INFO] {} Memory usage of data is {:.5f} MB.".format(
                str(datetime.now())[:-4], memory_usage_after_reduced))
            print("[INFO] Decreased by {:.4f}%.".format(
                100 * (memory_usage_before_transformed - \
                    memory_usage_after_reduced) \
                        / memory_usage_before_transformed))

        return self.data_table


class PurgedGroupTimeSeriesSplit:
    """针对带有Group id（组id）数据的时间序列交叉验证集合生成类。

    生成针对带有Group id的数据的时序交叉验证集。其中训练与验证的
    Group之间可以指定group_gap，用来隔离时间上的关系。这种情况下
    group_id通常是时间id，例如天或者小时。

    @Parameters:
    ----------
        n_splits: {int-like}, default=5
            切分的集合数目。
        max_train_group_size: {int-like}, default=+inf
            训练集单个组的最大样本数据限制。
        group_gap: {int-like}, default=None
            依据group_id切分组时，训练组与测试组的id的gap数目。
        max_test_group_size: {int-like}, default=+inf
            测试集单个组的最大样本数据限制。

    @References:
    ----------
    [1] https://www.kaggle.com/gogo827jz/jane-street-ffill-xgboost-purgedtimeseriescv
    """
    def __init__(self, n_splits=5, max_train_group_size=np.inf,
                 max_test_group_size=np.inf, group_gap=None, verbose=False):
        self.n_splits = n_splits
        self.max_train_group_size = max_train_group_size
        self.max_test_group_size = max_test_group_size
        self.group_gap = group_gap
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """生成训练组与测试组的id索引，返回组索引的生成器。

        @Parameters:
        ----------
            X: {array-like} {n_samples, n_features}
                训练数据，输入形状为{n_samples, n_features}。
            y: {array-like} {n_samples, }
                标签数据，形状为{n_samples, }。
            groups: {array-like} {n_samples, }
                用来依据组来划分训练集与测试集的组id，必须为连续的，有序的组id。

        @Yields:
        ----------
            train_idx: ndarray
                依据group_id切分的训练组id。
            test_idx: ndarray
                依据group_id切分的测试组id。
        """
        if X.shape[0] != groups.shape[0]:
            raise ValueError("The input shape mismatch!")

        # 构建基础参数
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size

        n_samples, n_splits, group_gap = len(X), self.n_splits, self.group_gap
        n_folds = n_splits - 1

        # 确定group_dict，用于存储每个组的样本index
        group_dict = {}
        unique_group_id, _ = np.unique(
            groups, return_index=True)

        # 扫描整个数据id list，构建group_dcit，{group_id: 属于该group的样本的idx}
        n_groups = len(unique_group_id)

        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]

        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds, n_groups))

        # test_group_size: 每个fold预留的test group的大小
        group_test_size = min(n_groups // n_splits, max_test_group_size)
        group_test_starts = range(n_groups - n_folds * group_test_size,
                                  n_groups, group_test_size)


        for group_test_start in group_test_starts:
            train_idx, gap_idx, test_idx = [], [], []

            # 计算train的group的起始位置
            group_train_start = max(0, group_test_start - \
                                       group_gap - max_train_group_size)

            for train_group_id in range(group_train_start,
                                        group_test_start - group_gap):
                raw_id = unique_group_id[train_group_id]
                if raw_id in group_dict:
                    train_idx.extend(group_dict[raw_id])

            for gap_id in range(group_test_start - group_gap,
                                group_test_start):
                raw_id = unique_group_id[gap_id]
                if raw_id in group_dict:
                    gap_idx.extend(group_dict[raw_id])

            for test_group_id in range(group_test_start,
                                       group_test_start + group_test_size):
                raw_id = unique_group_id[test_group_id]
                if raw_id in group_dict:
                    test_idx.extend(group_dict[raw_id])

            yield np.array(train_idx), np.array(gap_idx), np.array(test_idx)


class LiteModel:
    """将模型转换为Tensorflow Lite模型，提升推理速度。目前仅支持Keras模型转换。

    @Attributes:
    ----------
    interpreter: {Tensorflow lite transformed object}
        利用tf.lite.interpreter转换后的Keras模型。

    @References:
    ----------
    [1] https://medium.com/@micwurm/using-tensorflow-lite-to-speed-up-predictions-a3954886eb98
    """

    @classmethod
    def from_file(cls, model_path):
        """类方法。用于model_path下的模型，一般为*.h5模型。"""
        return LiteModel(tf.lite.Interpreter(model_path=model_path))

    @classmethod
    def from_keras_model(cls, kmodel):
        """类方法。用于直接转换keras模型。不用实例化类可直接调用该方法，返回
        被转换为tf.lite形式的Keras模型。

        @Attributes:
        ----------
        kmodel: {tf.keras model}
            待转换的Keras模型。

        @Returens:
        ----------
        经过转换的Keras模型。
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    def __init__(self, interpreter):
        """为经过tf.lite.interpreter转换的模型构建构造输入输出的关键参数。

        TODO(zhuoyin94@163.com):
        ----------
        [1] 可添加关键字，指定converter选择采用INT8量化还是混合精度量化。
        [2] 可添加关键字，指定converter选择量化的方式：低延迟还是高推理速度？
        """
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()

        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det["index"]
        self.output_index = output_det["index"]
        self.input_shape = input_det["shape"]
        self.output_shape = output_det["shape"]
        self.input_dtype = input_det["dtype"]
        self.output_dtype = output_det["dtype"]

    def predict(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i:i+1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.output_index)[0]
        return out

    def predict_single(self, inp):
        """ Like predict(), but only for a single record. The input data can be a Python list. """
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out[0]


@njit
def njit_infer_time2fault(
    time2fault_label_mat, min_fault_minutes=3, gap_minutes=5):
    """利用预先预测的time2fault矩阵，推断可能的time2fault时间。"""
    time2fault_array = np.zeros((len(time2fault_label_mat), 1))

    for row_idx in range(len(time2fault_label_mat)):
        label_row = time2fault_label_mat[row_idx, :]

        time2fault_minutes = min_fault_minutes
        for col_idx, label in enumerate(label_row):
            if label == 0:
                time2fault_minutes = \
                    min_fault_minutes + col_idx * gap_minutes
            else:
                break
        time2fault_array[row_idx] = time2fault_minutes
    return time2fault_array


def compute_custom_time2fault_metric(
    y_pred_time2fault, y_true_time2fault):
    """用于优化官方PTI与ATI的故障时间预测的回归Metric。"""
    y_pred_time2fault[y_pred_time2fault < 0] = 0

    eval_bool = np.where(
        (y_pred_time2fault > y_true_time2fault), 1, 0)
    y_true_time2fault = y_true_time2fault + 0.00000001

    y_pred_score = sigmoid(y_pred_time2fault / y_true_time2fault)
    y_pred_score[eval_bool == 1] = 0

    y_pred_score = np.mean(y_pred_score)

    return y_pred_score


@njit
def njit_f1(y_true, y_pred_proba, threshold):
    """通过阈值枚举搜索最优的F1的阈值, 采用@njit技术加速计算"""
    y_pred_label = np.where(y_pred_proba > threshold, 1, 0)

    # https://www.itread01.com/content/1544007604.html
    tp = np.sum(np.logical_and(np.equal(y_true, 1),
                               np.equal(y_pred_label, 1)))
    fp = np.sum(np.logical_and(np.equal(y_true, 0),
                               np.equal(y_pred_label, 1)))
    # tn = np.sum(np.logical_and(np.equal(y_true, 1),
    #                            np.equal(y_pred_label, 0)))
    fn = np.sum(np.logical_and(np.equal(y_true, 1),
                               np.equal(y_pred_label, 0)))

    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


@njit
def njit_window_group_count_diff(
        group_array, time_stamp_array,
        weight_array=None, time_window_size=10):
    """依据组标号与时间数组抽取给定feat_array的count统计量与count diff统计量"""
    if weight_array is None:
        weight_array = np.ones(len(group_array), dtype=np.int32)
    count_array = \
        np.zeros(int(len(time_stamp_array)), dtype=np.int32)
    count_shift_diff_array = \
        np.zeros(int(len(time_stamp_array)), dtype=np.int32)

    # 关键参数列表，if_first_exceed_range代表是否第一次超过time_window_size
    if_first_exceed_range = False
    curr_group_id, curr_window_count, array_size = None, 0, len(group_array)

    front = 0
    for rear in range(array_size):
        if curr_group_id is None:
            curr_group_id = group_array[rear]
        elif curr_group_id != group_array[rear]:
            curr_group_id = group_array[rear]
            curr_window_count = 0
            front = rear

        minutes_diff = (time_stamp_array[rear] - time_stamp_array[front]) / 60
        curr_window_count += weight_array[rear]

        # 若是从rear到front的时间不满足时间窗口的要求，则收缩窗口范围，直到满足要求为止
        if minutes_diff > time_window_size:
            if_first_exceed_range = True
            while(front < rear and minutes_diff > time_window_size):
                curr_window_count -= weight_array[front]
                front += 1
                minutes_diff = \
                    (time_stamp_array[rear] - time_stamp_array[front]) / 60

        # 计算diff的结果
        count_array[rear] = curr_window_count
        if if_first_exceed_range:
            count_shift_diff_array[rear] = \
                count_array[rear] - count_array[front]
        else:
            count_shift_diff_array[rear] = curr_window_count

    return count_array, count_shift_diff_array


@njit
def njit_window_group_sum_diff(
        group_array, time_stamp_array,
        feat_array, time_window_size=10):
    """依据time_stamp_array与group_array抽取给定观测窗口内的sum与sum_diff统计量"""
    feat_sum_diff_array = np.zeros(feat_array.shape, dtype=np.int32)
    feat_sum_array = np.zeros(feat_array.shape, dtype=np.int32)

    # 关键参数列表，if_first_exceed_range代表是否第一次超过time_window_size
    if_first_exceed_range = False
    curr_group_id, array_size = None, len(group_array)
    curr_window_sum = np.zeros(feat_array.shape[1])

    front = 0
    for rear in range(array_size):
        # 若curr_group_id发生变化，则重置curr_group_id与窗口内sum
        if curr_group_id is None:
            curr_group_id = group_array[rear]
        elif curr_group_id != group_array[rear]:
            curr_group_id = group_array[rear]
            curr_window_sum = np.zeros(feat_array.shape[1])
            front = rear

        # 累计窗口内sum
        minutes_diff = (time_stamp_array[rear] - time_stamp_array[front]) / 60
        curr_window_sum += feat_array[rear]

        # 若是从rear到front的时间不满足时间窗口的要求，则收缩窗口范围，直到满足要求为止
        if minutes_diff > time_window_size:
            if_first_exceed_range = True
            while(front < rear and minutes_diff > time_window_size):
                curr_window_sum -= feat_array[front]

                # 更新窗口时间信息
                front += 1
                minutes_diff = \
                    (time_stamp_array[rear] - time_stamp_array[front]) / 60

        # 计算diff的结果
        feat_sum_array[rear] = curr_window_sum
        if if_first_exceed_range:
            feat_sum_diff_array[rear] = \
                feat_sum_array[rear] - feat_sum_array[front]
        else:
            feat_sum_diff_array[rear] = curr_window_sum

    return feat_sum_array, feat_sum_diff_array


@njit
def njit_window_group_unique_count(
        group_array=None, time_stamp_array=None,
        feat_array=None, time_window_size=10):
    """依据time_stamp_array抽取时间窗口内的Unique count统计量"""
    feat_sum_array = np.zeros(feat_array.shape, dtype=np.int32)
    curr_group_id, array_size = None, len(group_array)
    curr_window_dict = {}

    front = 0
    for rear in range(array_size):
        # 若curr_group_id发生变化，则重置curr_group_id与窗口内sum
        if curr_group_id is None:
            curr_group_id = group_array[rear]
        elif curr_group_id != group_array[rear]:
            curr_group_id = group_array[rear]
            curr_window_dict.clear()
            front = rear

        # 累计窗口内unique count字典统计量
        minutes_diff = (time_stamp_array[rear] - time_stamp_array[front]) / 60
        if feat_array[rear] not in curr_window_dict:
            curr_window_dict[feat_array[rear]] = 1
        else:
            curr_window_dict[feat_array[rear]] += 1

        # 若是从rear到front的时间不满足时间窗口的要求，则收缩窗口范围，直到满足要求为止
        if minutes_diff > time_window_size:
            while(front < rear and minutes_diff > time_window_size):
                curr_window_dict[feat_array[front]] -=1

                if curr_window_dict[feat_array[front]] == 0:
                    curr_window_dict.pop(feat_array[front])

                # 更新窗口时间信息
                front += 1
                minutes_diff = \
                    (time_stamp_array[rear] - time_stamp_array[front]) / 60
        feat_sum_array[rear] = len(curr_window_dict)

    return feat_sum_array


@njit
def njit_window_group_std(
        group_array=None, time_stamp_array=None,
        feat_array=None, time_window_size=10):
    """依据time_stamp_array抽取时间窗口内feat_array的std统计量"""
    feat_std_array = np.ones(feat_array.shape, dtype=np.float32)

    curr_group_id, array_size = None, len(group_array)
    curr_cum_sum, curr_cum_square_sum, curr_window_count = 0.0, 0.0, 0.0

    front = 0
    for rear in range(array_size):
        # 若curr_group_id发生变化，则重置curr_group_id与窗口内统计量
        if curr_group_id is None:
            curr_group_id = group_array[rear]
        elif curr_group_id != group_array[rear]:
            curr_group_id = group_array[rear]
            curr_cum_sum, curr_cum_square_sum, curr_window_count = 0.0, 0.0, 0.0
            front = rear

        # 累计窗口内统计量
        minutes_diff = (time_stamp_array[rear] - time_stamp_array[front]) / 60
        curr_cum_sum += feat_array[rear]
        curr_cum_square_sum += np.square(feat_array[rear])
        curr_window_count += 1

        # 若是从rear到front的时间不满足时间窗口的要求，则收缩窗口范围，直到满足要求为止
        if minutes_diff > time_window_size:
            while(front < rear and minutes_diff > time_window_size):
                curr_cum_sum -= feat_array[front]
                curr_cum_square_sum -= np.square(feat_array[front])
                curr_window_count -= 1

                # 更新窗口时间信息
                front += 1
                minutes_diff = \
                    (time_stamp_array[rear] - time_stamp_array[front]) / 60

        # 计算与存储统计量结果
        feat_std_array[rear] = \
            curr_cum_square_sum / curr_window_count - \
                np.square(curr_cum_sum / curr_window_count)
    return feat_std_array


@njit
def njit_cut_array_to_bin(array, low=0, high=127, n_bins=0):
    """通过one-pass扫描与取余的方法，将array的内容切分到bin上去。"""
    binned_array = np.zeros(len(array), dtype=np.int32)
    range_per_bin = int((high - low + 1) / n_bins)

    # Invalid bin
    if low > high or low == high or range_per_bin == 0:
        return None

    for i, val in enumerate(array):
        val_bin_label, remainder = divmod(val, range_per_bin)

        if remainder != 0:
            val_bin_label += 1
        binned_array[i] = val_bin_label
    return binned_array


@njit
def make_prediction(group_array=None, time_stamp_array=None):
    """对于所有预测为1的按照sn与time_stamp由小到大排序的数据，生成预测bool数组。"""
    is_valid_sub_array = np.ones((len(group_array), )) == 0

    # 生成需要提交的bool index array
    front, curr_group_id = 0, None
    for rear in range(len(group_array)):
        if curr_group_id is None:
            is_valid_sub_array[rear] = True
            curr_group_id = group_array[rear]
        elif curr_group_id != group_array[rear]:
            is_valid_sub_array[rear] = True
            curr_group_id = group_array[rear]
            front = rear

        # 距离上一次提交的时间是否超过7天
        minutes_diff = (time_stamp_array[rear] - \
            time_stamp_array[front]) / 60
        if minutes_diff >= 10080:
            is_valid_sub_array[rear] = True
            front = rear

    return is_valid_sub_array


@njit
def sigmoid(x):
    """sigmoid函数实现。"""
    return np.exp(-np.logaddexp(0, -x))


@njit
def make_evaluation(sn_array=None,
                    pred_failure_date_array=None,
                    pred_time2fault_array=None,
                    failure_tag_dict=None):
    """针对给定的预测failure时间，计算该结果对应的官方Metric的分数"""
    # 预测标签的数组
    y_pred_labels = np.zeros((len(sn_array))) - 1
    y_pred_pti = np.zeros((len(sn_array))) - 1
    y_true_ati = np.zeros((len(sn_array))) - 1

    # 计算每一个预测的时间是否落在对应的failure time区间内
    for i in range(len(sn_array)):
        # 获取当前的serial_number与当前的预测窗口left edge时间
        curr_sn = sn_array[i]
        curr_pred_date = pred_failure_date_array[i]

        # 若该sn位于failure_tag_dict中
        if curr_sn in failure_tag_dict and \
            curr_pred_date > failure_tag_dict[curr_sn][0] and \
                curr_pred_date < failure_tag_dict[curr_sn][1]:
            y_pred_labels[i] = 1
            y_pred_pti[i] = \
                pred_time2fault_array[i]

            # date diff(seconds)
            y_true_ati[i] = \
                (failure_tag_dict[curr_sn][1] - curr_pred_date) / 60
        else:
            y_pred_labels[i] = 0

    # npr: 评估窗口内所有发生内存故障服务器的数量
    npr = len(failure_tag_dict)

    # ntpr: 评估窗口内发生内存故障的服务器被提前7天内发现的数量
    y_pred_pti_tmp = y_pred_pti[y_pred_pti >= 0]
    y_true_ati_tmp = y_true_ati[y_pred_pti >= 0]

    ntpr = 0
    for idx in range(len(y_pred_pti_tmp)):
        if y_pred_pti_tmp[idx] <= y_true_ati_tmp[idx]:
            ntpr += sigmoid(y_pred_pti_tmp[idx] / y_true_ati_tmp[idx])

    # 召回率
    if npr == 0:
        recall = 0
    else:
        recall = ntpr / npr

    # npp: 评估窗口内被预测出来未来7天会发生故障的服务器数量
    npp = len(y_pred_labels)

    # ntpp: 评估窗口内第一次预测故障时间后7天内确实发生故障的服务器数量(与ntpr一致)
    ntpp = ntpr

    # 精准率
    if npp == 0:
        precision = 0
    else:
        precision = ntpp / npp

    # f1分数
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall


def make_df_evaluation(df=None):
    """评估df对应的Custom Metric计算的分数，并返回评估结果。

    df中必须拥有以下columns:
    - serial_number: {int-like}，预测的序列号，为int型。
    - y_pred_label: {int-like}，预测的故障样本标签。
    - y_pred_time2fault： {float-like}，预测的该样本距离故障的时间，分钟为单位。
    - collect_time_bin_right_edge： {int-like}，预测故障样本的右观测边界。
    """
    if ("y_pred_label" not in df.columns) or ("y_true" not in df.columns):
        raise ValueError("Missing vital columns !")

    if ("collect_time_bin_right_edge" not in df.columns) or \
        ("y_pred_time2fault" not in df.columns):
        raise ValueError("Missing vital columns !")

    # 载入所有failure数据集，并按时间顺序进行排序
    # -------------------
    file_processor = LoadSave(verbose=0)
    failure_tag = file_processor.load_data(dir_name="./data_tmp/",
        file_name="failure_tag.pkl")
    failure_tag["min_range"] = (failure_tag["failure_time"] + \
                                pd.Timedelta(-7, unit="d")).astype(int) / 10**9
    failure_tag["max_range"] = failure_tag["failure_time_unix"].values

    df["serial_number"] = df["serial_number"].astype(int)
    df["collect_time_bin_right_edge"] = \
        df["collect_time_bin_right_edge"].astype(int)

    # df预测范围内的failure tag的相关信息
    failure_tag_valid = failure_tag[
        (failure_tag["failure_time_unix"] > \
         df["collect_time_bin_right_edge"].min()) & \
        (failure_tag["failure_time_unix"] < \
         df["collect_time_bin_right_edge"].max())].reset_index(drop=True)

    # 使用numba的Dict对象进行封装，方便进行评估
    failure_tag_valid_dict = numba.typed.Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.float64[:])
    for idx in range(len(failure_tag_valid)):
        sn_number = failure_tag_valid.iloc[idx]["serial_number"]
        failure_tag_valid_dict[sn_number] = \
            np.array([failure_tag_valid.iloc[idx]["min_range"],
                      failure_tag_valid.iloc[idx]["max_range"]],
                      dtype=np.float64)

    # 开始评估df的分数
    # - 按时间排序df的值
    # - 挑选标签为1的样本
    # - 调用评估方法对分数进行计算
    # -------------------
    df.sort_values(
        by=["serial_number", "collect_time_bin_right_edge"],
        ascending=[True, True], inplace=True)

    tmp_df = df.query("y_pred_label == 1")
    tmp_df = tmp_df.groupby(["serial_number"]).agg(
        "first").reset_index()

    f1_res, precision_res, recall_res = make_evaluation(
        tmp_df["serial_number"].values,
        tmp_df["collect_time_bin_right_edge"].values,
        tmp_df["y_pred_time2fault"].values,
        failure_tag_valid_dict)
    return f1_res, precision_res, recall_res, len(tmp_df)/len(failure_tag_valid)
