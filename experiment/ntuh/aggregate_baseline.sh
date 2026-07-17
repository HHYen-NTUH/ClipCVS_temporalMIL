#!/bin/bash

set -e

ROOTDIR="<repository directory>"
PREDDIR="${ROOTDIR}/analysis/ntucholecvs/evaluation"
PARENTNAME="ntuh_lstm"
PARENTRUN=1
NAME="ntuh_temporal_mil"
RUN=2
AGGMETHOD="mean_logit"

python aggregate_clip_prediction.py \
  --srcpredpath "${PREDDIR}/${PARENTNAME}/run_${PARENTRUN}/prediction.csv" \
  --dstpredpath "${PREDDIR}/${NAME}/run_${RUN}/prediction.csv" \
  --aggmethod ${AGGMETHOD} \
  --aggpredpath "${PREDDIR}/${NAME}/run_${RUN}/swincvs_aggregate/${AGGMETHOD}/prediction.csv"

python evaluate_prediction.py \
  --predpath "${PREDDIR}/${NAME}/run_${RUN}/swincvs_aggregate/${AGGMETHOD}/prediction.csv" \
  --evaldir "${PREDDIR}/${NAME}/run_${RUN}/swincvs_aggregate/${AGGMETHOD}"
