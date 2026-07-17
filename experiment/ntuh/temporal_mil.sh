#!/bin/bash

set -e

ROOTDIR="<repository directory>"
DATADIR="<directory of NTUH dataset>"
PARENTNAME="ntuh_swinv2"
PARENTRUN=1
NAME="ntuh_temporal_mil"
MODEL="Gated attention MIL"
RUN=2
SEQLEN=30
LBLAGG="more_than_50%"
DESC_SUFFIX="temporal model of ${SEQLEN}-frame clips with ${LBLAGG} aggregation"

python train_flexible_temporal_mil.py \
  --dataset "ntucholecvs" \
  --datadir "${DATADIR}/raw" \
  --metadatadir "${DATADIR}/artifact/metadata" \
  --videometapath "${DATADIR}/artifact/metadata/Document03_ annotation summary of NTUcholeCVS videos.csv" \
  --seqlen ${SEQLEN} \
  --lblagg_method ${LBLAGG} \
  --featdir "${ROOTDIR}/artifact/ntucholecvs/frame_feature/${PARENTNAME}/run_${PARENTRUN}" \
  --stack_frame_feats \
  --featsize 1024 \
  --modeltype "${MODEL}" \
  --hidden_size 256 \
  --dropout 0.1 \
  --label_weights 2.5362 5.1090 1.8576 \
  --learning_rate 0.00125 \
  --weight_decay 0.05 \
  --use_warmup_cosine_scheduler \
  --num_warmup_epochs 5 \
  --batchsize 32 \
  --num_workers 2 \
  --device "cuda:0" \
  --num_epochs 20 \
  --logfreq 25 \
  --mlflow_server_uri "file:${ROOTDIR}/logs/mlflow/mlruns" \
  --mlflow_expname "${NAME}" \
  --mlflow_runname "run_${RUN}" \
  --mlflow_rundesc "${MODEL} ${DESC_SUFFIX}" \
  --ckptdir "${ROOTDIR}/model/${NAME}/run_${RUN}" \
  --ckptname "best_ckpt.pt"

python predict_flexible_temporal_mil.py \
  --dataset "ntucholecvs" \
  --datadir "${DATADIR}/raw" \
  --metadatadir "${DATADIR}/artifact/metadata" \
  --videometapath "${DATADIR}/artifact/metadata/Document03_ annotation summary of NTUcholeCVS videos.csv" \
  --seqlen ${SEQLEN} \
  --lblagg_method ${LBLAGG} \
  --featdir "${ROOTDIR}/artifact/ntucholecvs/frame_feature/${PARENTNAME}/run_${PARENTRUN}" \
  --stack_frame_feats \
  --featsize 1024 \
  --modeltype "${MODEL}" \
  --hidden_size 256 \
  --ckptpath "${ROOTDIR}/model/${NAME}/run_${RUN}/best_ckpt.pt" \
  --batchsize 32 \
  --num_workers 2 \
  --device "cuda:0" \
  --predpath "${ROOTDIR}/analysis/ntucholecvs/evaluation/${NAME}/run_${RUN}/prediction.csv"

python evaluate_prediction.py \
  --predpath "${ROOTDIR}/analysis/ntucholecvs/evaluation/${NAME}/run_${RUN}/prediction.csv" \
  --evaldir "${ROOTDIR}/analysis/ntucholecvs/evaluation/${NAME}/run_${RUN}"
