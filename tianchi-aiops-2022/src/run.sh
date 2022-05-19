# 数据进行预处理
python preprocessing.py

# 训练msg embedding：(joint与split形式进行训练)
python train_msg_embedding.py \
    --cbow-n-iters 8 \
    --sg-n-iters 5 \
    --window-size 16 \
    --embedding-dim 40 \
    --min-count 1 \
    --style joint

python train_msg_embedding.py \
    --cbow-n-iters 8 \
    --sg-n-iters 5 \
    --window-size 16 \
    --embedding-dim 40 \
    --min-count 1 \
    --style split

# 抽取msg模板信息
python compute_log_template.py \
    --keep-top-n-template 220

# 训练template id的embedding
python train_template_embedding.py \
    --window-size 16 \
    --embedding-dim 16

# lgb特征工程
python compute_fe_lgb.py

# 训练lgb模型（定论次集成）
python train_lgb_v2.py \
    --n-folds 5 \
    --sub-style online \
    --n-estimators 7000 \
    --early-stopping-rounds 1000 \
    --keep-last-k-days 40 \
    --cv-strategy gkf

# 压缩result.csv
zip -rq result.zip result.csv
rm result.csv
