#!/bin/bash

set -e

ROOTDIR="<repository directory>"
DATADIR="<directory of Endoscapes dataset>"
NAME="endoscapes_lstm"
RUN=1
DESC="Default configuration"

python train_flexible_lstm.py \
  --dataset "endoscapes" \
  --datadir "${DATADIR}" \
  --seqlen 5 \
  --lblagg_method "last" \
  --featdir "${ROOTDIR}/artifact/endoscapes/frame_feature/pretrained_swincvs_frozen_backbone" \
  --stack_frame_feats \
  --featsize 1024 \
  --learning_rate 0.001 \
  --weight_decay 0.02 \
  --label_weights 5.3971 7.9231 4.5904 \
  --batchsize 32 \
  --num_workers 2 \
  --device "cuda:0" \
  --num_epochs 20 \
  --logfreq 25 \
  --mlflow_server_uri "file:${ROOTDIR}/logs/mlflow/mlruns" \
  --mlflow_expname "${NAME}" \
  --mlflow_runname "run_${RUN}" \
  --mlflow_rundesc "${DESC}" \
  --ckptdir "${ROOTDIR}/model/${NAME}/run_${RUN}" \
  --ckptname "best_ckpt.pt"

python predict_flexible_lstm.py \
  --dataset "endoscapes" \
  --datadir "${DATADIR}" \
  --seqlen 5 \
  --lblagg_method "last" \
  --featdir "${ROOTDIR}/artifact/endoscapes/frame_feature/pretrained_swincvs_frozen_backbone" \
  --stack_frame_feats \
  --featsize 1024 \
  --ckptpath "${ROOTDIR}/model/${NAME}/run_${RUN}/best_ckpt.pt" \
  --batchsize 32 \
  --num_workers 2 \
  --device "cuda:0" \
  --predpath "${ROOTDIR}/analysis/endoscapes/evaluation/${NAME}/run_${RUN}/prediction.csv"

python evaluate_prediction.py \
  --predpath "${ROOTDIR}/analysis/endoscapes/evaluation/${NAME}/run_${RUN}/prediction.csv" \
  --evaldir "${ROOTDIR}/analysis/endoscapes/evaluation/${NAME}/run_${RUN}"
