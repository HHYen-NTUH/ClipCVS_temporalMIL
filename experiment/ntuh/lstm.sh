#!/bin/bash

set -e

ROOTDIR="<repository directory>"
DATADIR="<directory of NTUH dataset>"
PARENTNAME="ntuh_swinv2"
PARENTRUN=1
NAME="ntuh_lstm"
RUN=1
DESC="Default configuration"

python train_flexible_lstm.py \
  --dataset "ntucholecvs" \
  --datadir "${DATADIR}/raw" \
  --metadatadir "${DATADIR}/artifact/metadata" \
  --videometapath "${DATADIR}/artifact/metadata/Document03_ annotation summary of NTUcholeCVS videos.csv" \
  --seqlen 5 \
  --lblagg_method "last" \
  --featdir "${ROOTDIR}/artifact/ntucholecvs/frame_feature/${PARENTNAME}/run_${PARENTRUN}" \
  --stack_frame_feats \
  --featsize 1024 \
  --label_weights 2.5362 5.1090 1.8576 \
  --learning_rate 0.001 \
  --weight_decay 0.02 \
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
  --dataset "ntucholecvs" \
  --datadir "${DATADIR}/raw" \
  --metadatadir "${DATADIR}/artifact/metadata" \
  --videometapath "${DATADIR}/artifact/metadata/Document03_ annotation summary of NTUcholeCVS videos.csv" \
  --seqlen 5 \
  --lblagg_method "last" \
  --featdir "${ROOTDIR}/artifact/ntucholecvs/frame_feature/${PARENTNAME}/run_${PARENTRUN}" \
  --stack_frame_feats \
  --featsize 1024 \
  --ckptpath "${ROOTDIR}/model/${NAME}/run_${RUN}/best_ckpt.pt" \
  --batchsize 32 \
  --num_workers 2 \
  --device "cuda:0" \
  --predpath "${ROOTDIR}/analysis/ntucholecvs/evaluation/${NAME}/run_${RUN}/prediction.csv"

python evaluate_prediction.py \
  --predpath "${ROOTDIR}/analysis/ntucholecvs/evaluation/${NAME}/run_${RUN}/prediction.csv" \
  --evaldir "${ROOTDIR}/analysis/ntucholecvs/evaluation/${NAME}/run_${RUN}"
