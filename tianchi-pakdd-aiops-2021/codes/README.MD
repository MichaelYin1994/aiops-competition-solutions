## PAKDD 2021 && 阿里云智能运维算法大赛论文相关源代码

---
### **代码说明**

具体来说本论文的源代码包括4个不同的部分：预处理阶段，特征工程阶段，集成召回模型训练阶段与故障预测模型训练阶段：
- **预处理阶段**：主要为**offline_read_processing.py**模块。该模块的主要功能为读入初赛数据，并进行类型转换，使用-1填充np.nan缺失值，处理str字符串特征等工作，不涉及任何时间序列上的操作。
- **特征工程阶段**：分别为**air_compute_feats_address_log.py**，**air_compute_feats_kernel_log.py**，**air_compute_feats_mce_log.py**与**offline_compute_feats_concat.py**部分。其中**offline_compute_feats_concat.py**是将其他各组日志的特征文件按观测窗口ID与Serial Number按键进行聚合。其他的模块分别对各自对应的日志进行特征工程，所有日志的特征工程都具有一定的相似性，分为以下两大类特征：
    - 一般性日志特征：统计每一时刻，每一条日志给定历史窗口范围例如3分钟、30分钟、120分钟内的日志数量，然后按照从2019-01-01 00:00:00开始算起给定观测窗口（例如观测窗口可以是20秒或者1分钟）的ID信息进行聚合，提取每个观测窗口上的统计量及其按时间进行shift的特征。并且记录每一个观测窗口的边界信息。
    - 日志专家特征：例如针对row与col信息而言，抽取给定时间窗口内的例如3分钟、30分钟、120分钟内的unique的样本个数等。
- **集成召回模型训练阶段**：为**model_offline_xgb_recall.py**模块。依据前述方法训练集成召回模型，选择最优的切分阈值。并预测每一个样本的召回概率，依据召回概率结果滤除信息量不大的样本。
- **故障检测模型阶段**：为**model_offline_xgb_reg.py**模块。固定阈值法训练故障时间回归模型，详细信息见前述原理说明与代码注释。
- **故障时间预测模型**：为**model_offline_nn_time2fault.py**模块。构造了残差结构的Multi-label Neural Network，基于Multi-label的方法对故障时间进行预测，减小Overestimate的风险。
- **其他辅助文件**：**utils.py**，内部包含经过高度优化的各种Metric计算方法，特征提取方法与一些数据处理与数据分析的辅助工具。**test_create_stream_log.py**，这是比赛中设计的一个测试脚本，由于线上需要进行流推理，因此需要在线下尽量模拟线上环境进行测试。该脚本对线下数据进行预处理，变换为线上数据类似形式，方便进行流测试。

---
### **代码文件结构**

本项目实际代码结构（源码内部按此目录结构撰写）：
```
.
├── README.MD
├── compute_feats_address_log_xgb.py
├── compute_feats_concat_xgb.py
├── compute_feats_kernel_log_xgb.py
├── compute_feats_mce_log_xgb.py
├── ./data/
├── ./data_tmp/
├── model_recall_stage_clf_ensemble.py
├── model_xgb_reg_ensemble.py
├── ./models/
├── read_processing.py
├── ./submissions/
└── utils.py
```

目录功能说明：
- **./data/**：用于存储解压后的无二级目录的原始*.csv数据。**read_processing.py**将会从该目录读取原始*.csv文件进行预处理，并将处理结果存在./data_tmp/目录下。
- **./data_tmp/**：用于预处理后的数据与特征工程之后的特征数据。
- **./models/**：用于存储召回模型以及召回模型预测的召回概率。
- **./submissions/**：用于存储回归模型预测生成的submission文件。
