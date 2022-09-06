#!/bin/bash
#SBATCH -J d_score_test                              # 作业名为 test
#SBATCH -o d_score_t.out                      # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 20:00:00                           # 任务运行的最长时间为 20 小时
#SBATCH -w gpu28                              # 指定运行作业的节点是 gpu06，若不填写系统自动分配节点
#SBATCH --gres=gpu:a100-pcie-40gb:1

# Variables
GPU=0
SEED=71
DATASET=dailydialog_plusplus_mlr
DATA=./dataset/dailydialog++
SPLIT2FOUR=False
MODEL=roberta_metric
TRAINER=mlr_pretrain
MODE=train
TRAIN_MODE=medium

TRAIN_BATCH_SIZE=1
EVAL_BATCH_SIZE=1
TEST_BATCH_SIZE=1

NUM_EPOCHS=5
LEARNING_RATE=2e-5
WARMUP_PROPORTION=0.1
FEATURE_DISTANCE_LOWER_BOUND=1
FEATURE_DISTANCE_UPPER_BOUND=0.1
SCORE_DISTANCE_LOWER_BOUND=[0.6,0.3,0.15]
SCORE_DISTANCE_UPPER_BOUND=0.1
FEATURE_LOSS_WEIGHT=0.
SCORE_LOSS_WEIGHT=1.
BCE_LOSS_WEIGHT=0.
MONITOR_METRIC_NAME=dual_mlr_loss
MONITOR_METRIC_TYPE=min
CHECKPOINT_DIR_PATH=./output/$SEED/roberta_pretrain
LOGGING_LEVEL=INFO
CENTROID_MODE=mean
DISTANCE_MODE=cosine
PRETRAINED_MODEL_NAME=roberta-large


# Returns to main directory

# Train
python pretrain.py \
    --data_path $DATA \
    --split2four $SPLIT2FOUR\
    --train_mode $TRAIN_MODE\
    --dataset $DATASET \
    --model $MODEL \
    --trainer $TRAINER \
    --mode $MODE \
    --gpu $GPU \
    --checkpoint_dir_path $CHECKPOINT_DIR_PATH \
    --seed $SEED \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --test_batch_size $TEST_BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --monitor_metric_name $MONITOR_METRIC_NAME \
    --monitor_metric_type $MONITOR_METRIC_TYPE \
    --feature_distance_lower_bound $FEATURE_DISTANCE_LOWER_BOUND \
    --feature_distance_upper_bound $FEATURE_DISTANCE_UPPER_BOUND \
    --score_distance_lower_bound $SCORE_DISTANCE_LOWER_BOUND \
    --score_distance_upper_bound $SCORE_DISTANCE_UPPER_BOUND \
    --feature_loss_weight $FEATURE_LOSS_WEIGHT \
    --score_loss_weight $SCORE_LOSS_WEIGHT \
    --bce_loss_weight $BCE_LOSS_WEIGHT \
    --learning_rate $LEARNING_RATE \
    --warmup_proportion $WARMUP_PROPORTION \
    --logging_level $LOGGING_LEVEL \
    --centroid_mode $CENTROID_MODE \
    --distance_mode $DISTANCE_MODE \
    --pretrained_model_name $PRETRAINED_MODEL_NAME \
    --weighted_s_loss