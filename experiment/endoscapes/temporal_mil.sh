#!/bin/bash

set -e

ROOTDIR="<repository directory>"
DATADIR="<directory of Endoscapes dataset>"
NAME="endoscapes_temporal_mil"

SEQLENS=(
  10
  30
  60
  90
)
MODELS=(
  "Mean MIL"
  "Max MIL"
  "Gated attention MIL"
  "Self attention MIL"
)
LBLAGGS=(
  "more_than_50%"
  "union"
)

run=1
for lblagg in "${LBLAGGS[@]}"; do
  for seqlen in "${SEQLENS[@]}"; do
    for model in "${MODELS[@]}"; do
      ((run++))
      desc="Run ${run}: ${model} temporal model of ${seqlen}-frame clips with ${lblagg} aggregation"
      echo "${desc}"

      python train_flexible_temporal_mil.py \
        --dataset "endoscapes" \
        --datadir "${DATADIR}" \
        --seqlen ${seqlen} \
        --lblagg_method ${lblagg} \
        --featdir "${ROOTDIR}/artifact/endoscapes/frame_feature/pretrained_swincvs_frozen_backbone" \
        --stack_frame_feats \
        --featsize 1024 \
        --modeltype "${model}" \
        --hidden_size 256 \
        --dropout 0.1 \
        --label_weights 5.3971 7.9231 4.5904 \
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
        --mlflow_runname "run_${run}" \
        --mlflow_rundesc "${desc}" \
        --ckptdir "${ROOTDIR}/model/${NAME}/run_${run}" \
        --ckptname "best_ckpt.pt"

      python predict_flexible_temporal_mil.py \
        --dataset "endoscapes" \
        --datadir "${DATADIR}" \
        --seqlen ${seqlen} \
        --lblagg_method ${lblagg} \
        --featdir "${ROOTDIR}/artifact/endoscapes/frame_feature/pretrained_swincvs_frozen_backbone" \
        --stack_frame_feats \
        --featsize 1024 \
        --modeltype "${model}" \
        --hidden_size 256 \
        --ckptpath "${ROOTDIR}/model/${NAME}/run_${run}/best_ckpt.pt" \
        --batchsize 32 \
        --num_workers 2 \
        --device "cuda:0" \
        --predpath "${ROOTDIR}/analysis/endoscapes/evaluation/${NAME}/run_${run}/prediction.csv"

      python evaluate_prediction.py \
        --predpath "${ROOTDIR}/analysis/endoscapes/evaluation/${NAME}/run_${run}/prediction.csv" \
        --evaldir "${ROOTDIR}/analysis/endoscapes/evaluation/${NAME}/run_${run}"
    done
  done
done
