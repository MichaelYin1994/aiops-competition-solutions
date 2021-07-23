## PAKDD 2021 && 阿里云智能运维算法大赛

---
### **队伍简介**

队名：OutOfMemory，队长：鱼丸粗面（zhuoyin94@163.com）。初赛A榜2/1350，初赛Metric的F1值62.37；初赛B榜2/1350，初赛Metric的F1值为64.52；复赛代码上线工业级流水线测评环境，复赛A榜12/1350，分数为30.95，复赛B轮排名7/1350，分数28.63；综合答辩排名5/1350，获得优胜奖与PAKDD 2021 workshop最佳论文奖。整体采用了[此项目](https://github.com/MichaelYin1994/python-style-guide)的编码规范。（20210204）

~~**应主办方要求，本方案代码暂不开源。若后续有开源计划，请移步此[Github开源仓库](https://github.com/MichaelYin1994/tianchi-pakdd-aiops-2021)。**~~

**20210723，经过主办方同意，开源论文部分相关代码。包括论文数据预处理部分，特征工程部分，Two-stage策略部分与基于Multi-label的故障时间预测部分。详见codes文件夹。**

---
### **系统环境与主要依赖packages**

- 系统环境: Ubuntu 18.04 LTS
- GPU: Quadro P5000(16G)
- CPU: Intel® Core™ i9-9820X CPU @ 3.30GHz × 20
- RAM: 32G
- CUDA: 10.1
- swap: 64G
- python == 3.6.9
- pandas == 1.0.4
- numba == 0.49.0
- numpy == 1.18.1
- xgboost == 1.2.1
- scikit-learn == 0.22.2
- scipy == 1.4.1
- tensorflow-gpu == 2.1.0

---
### **文档，答辩录播与阿里云数据开源地址**

文档下载备用链接：
- 整体方案说明、PAKDD 2021 Workshop汇报PPT与线下答辩PPT百度网盘下载链接：https://pan.baidu.com/s/1DZz0AjM1-DskhOUVi2SvOg
- 文件提取码：374s
- PAKDD 2021 AIOps workshop线上会答辩录屏：https://www.bilibili.com/video/BV1gf4y1h7vY/
- PAKDD 2021 & 第二届阿里云智能运维线下答辩录播： https://tianchi.aliyun.com/course/live/1654?spm=5176.21852598.0.0.209e37e8fiR3Nn
- 阿里云数据开源计划项目地址：https://github.com/alibaba-edu/dcbrain

---
### **References**

[1] Covington, Paul, Jay Adams, and Emre Sargin. "Deep neural networks for youtube recommendations." Proceedings of the 10th ACM conference on recommender systems. 2016.

[2] Du, Xiaoming, et al. "Predicting Uncorrectable Memory Errors for Proactive Replacement: An Empirical Study on Large-Scale Field Data." 2020 16th European Dependable Computing Conference (EDCC). IEEE, 2020.

[3] Schroeder, Bianca, Eduardo Pinheiro, and Wolf-Dietrich Weber. "DRAM errors in the wild: a large-scale field study." ACM SIGMETRICS Performance Evaluation Review 37.1 (2009): 193-204.

[4] Du, Xiaoming, and Cong Li. "Memory failure prediction using online learning." Proceedings of the International Symposium on Memory Systems. 2018.

[5] Hwang, Andy A., Ioan A. Stefanovici, and Bianca Schroeder. "Cosmic rays don't strike twice: Understanding the nature of DRAM errors and the implications for system design." ACM SIGPLAN Notices 47.4 (2012): 111-122.

[6] Sridharan, Vilas, and Dean Liberty. "A study of DRAM failures in the field." SC'12: Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis. IEEE, 2012.

[7] Boixaderas, Isaac, et al. "Cost-aware prediction of uncorrected DRAM errors in the field." SC20: International Conference for High Performance Computing, Networking, Storage and Analysis. IEEE, 2020.

[8] Lam, Siu Kwan, Antoine Pitrou, and Stanley Seibert. "Numba: A llvm-based python jit compiler." Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC. 2015.

[9] Chen, Tianqi, and Carlos Guestrin. "Xgboost: A scalable tree boosting system." Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining. 2016.

[10] Zhang, Huan, Si Si, and Cho-Jui Hsieh. "GPU-acceleration for Large-scale Tree Boosting." arXiv preprint arXiv:1706.08359 (2017).

[11] https://github.com/MichaelYin1994/python-style-guide

[12] [PAKDD 2020 && 阿里云磁盘寿命预测比赛冠军方案](https://tianchi.aliyun.com/forum/postDetail?postId=107912)

[13] Ma M, Zhang S, Pei D, et al. Robust and rapid adaption for concept drift in software system anomaly detection[C]//2018 IEEE 29th International Symposium on Software Reliability Engineering (ISSRE). IEEE, 2018: 13-24.

[14] Liu F T, Ting K M, Zhou Z H. Isolation forest[C]//2008 eighth ieee international conference on data mining. IEEE, 2008: 413-422.

[15] Goldstein M, Dengel A. Histogram-based outlier score (hbos): A fast unsupervised anomaly detection algorithm[J]. KI-2012: Poster and Demo Track, 2012: 59-63.
