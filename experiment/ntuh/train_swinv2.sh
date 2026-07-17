#!/bin/bash

set -e

ROOTDIR="<repository directory>"
DATADIR="<directory of NTUH dataset>"
NAME="ntuh_swinv2"
RUN=1
DESC="Default configuration"

python train_swinv2.py \
  --dataset "ntucholecvs" \
  --datadir "${DATADIR}/raw" \
  --metadatadir "${DATADIR}/artifact/metadata" \
  --videometapath "${DATADIR}/artifact/metadata/Document03_ annotation summary of NTUcholeCVS videos.csv" \
  --disable_minmaxscale \
  --swinv2_ckptpath "${ROOTDIR}/pretrained/swincvs/Swin_backbone_no_augm_sd4_bestMAP.pt" \
  --learning_rate 0.00001 \
  --weight_decay 0.02 \
  --label_weights 2.5362 5.1090 1.8576 \
  --batchsize 4 \
  --num_workers 2 \
  --device "cuda:0" \
  --num_epochs 100 \
  --num_accml_batches 8 \
  --logfreq 25 \
  --mlflow_server_uri "file:${ROOTDIR}/logs/mlflow/mlruns" \
  --mlflow_expname "${NAME}" \
  --mlflow_runname "run_${RUN}" \
  --mlflow_rundesc "${DESC}" \
  --ckptdir "${ROOTDIR}/model/${NAME}/run_${RUN}" \
  --ckptname "best_ckpt.pt"
