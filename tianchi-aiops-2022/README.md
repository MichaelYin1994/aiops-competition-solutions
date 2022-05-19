## 第三届阿里云磐久智维算法大赛

202202260033

### 运行

第一步，预处理：

```
python preprocessing.py
```

第二步，训练词向量：

```
python train_embedding.py
```

第三步，特征工程，只有两部分：TFIDF特征 + 句子mean word embedding：

```
python compute_fe_xgb.py
```

第四步，模型训练：

```
python train_xgb.py
```


### 可能的思路

- 频繁模式挖掘算法应用。
- ~~日志模板提取算法应用。~~
- ~~同义词挖掘算法应用（同义词聚类算法）。~~
- 多样化的Embedding，例如Glove + FastText策略（Glove的Tensorflow 2.x实现）。

- 改变滑窗的策略，控制滑动窗口内的样本的数目。
- ~~针对某些具体特征的target encoding策略。~~
- 使用nn时，采用数据增强策略。
- Deep learning加入新的sequence（例如time to fault的sequence）

- unique event type/ log msg count, most frequent log event template(category)
- ~~按窗口大小进行ensemble~~
- ~~CPU0C0_DIMM_Stat 这样的信息进行更精细化处理~~
- ~~HEX抽取前1到2个字符串，然后搞事情~~
- ~~msg log按 | 拆分开的unique count~~
- ~~log template id tf-idf~~
- ~~unique template id~~
- 窗口内event按时间分bin count的mean std
- meta_dict groupby time_id agg event count
- ~~干additional log的语料~~
- ~~随机删除一些数据做数据增强（或者随机采样一些日志数据）~~
- ~~时间上TFIDF与时间上的embedding特征~~
- ~~伪标签~~
- 不同window size的embedding

### References

---

[1] He, Pinjia, Jieming Zhu, Zibin Zheng, and Michael R. Lyu. "Drain: An online log parsing approach with fixed depth tree." In 2017 IEEE international conference on web services (ICWS), pp. 33-40. IEEE, 2017.

[2] Du, Min, Feifei Li, Guineng Zheng, and Vivek Srikumar. "Deeplog: Anomaly detection and diagnosis from system logs through deep learning." In Proceedings of the 2017 ACM SIGSAC conference on computer and communications security, pp. 1285-1298. 2017.

[3] He, Shilin, Jieming Zhu, Pinjia He, and Michael R. Lyu. "Experience report: System log analysis for anomaly detection." In 2016 IEEE 27th international symposium on software reliability engineering (ISSRE), pp. 207-218. IEEE, 2016.

[4] Du, Min, and Feifei Li. "Spell: Streaming parsing of system event logs." In 2016 IEEE 16th International Conference on Data Mining (ICDM), pp. 859-864. IEEE, 2016.

[5] https://www.usenix.org/legacy/publications/library/proceedings/usenix02/tech/freenix/full_papers/watanabe/watanabe_html/node6.html

[6] https://docs.microsoft.com/en-us/windows/win32/power/system-power-states

[7] https://github.com/logpai/logparser

[8] https://github.com/IBM/Drain3

[9] He, Shilin, Pinjia He, Zhuangbin Chen, Tianyi Yang, Yuxin Su, and Michael R. Lyu. "A survey on automated log analysis for reliability engineering." ACM Computing Surveys (CSUR) 54, no. 6 (2021): 1-37.

[10] Meng, Weibin, Ying Liu, Yichen Zhu, Shenglin Zhang, Dan Pei, Yuqing Liu, Yihao Chen et al. "LogAnomaly: Unsupervised detection of sequential and quantitative anomalies in unstructured logs." In IJCAI, vol. 19, no. 7, pp. 4739-4745. 2019.

[11] Lin, Qingwei, Hongyu Zhang, Jian-Guang Lou, Yu Zhang, and Xuewei Chen. "Log clustering based problem identification for online service systems." In Proceedings of the 38th International Conference on Software Engineering Companion, pp. 102-111. 2016.

[12] https://github.com/GradySimon/tensorflow-glove
