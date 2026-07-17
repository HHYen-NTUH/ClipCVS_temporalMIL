#!/bin/bash

set -e

ROOTDIR="<repository directory>"
DATADIR="<directory of NTUH dataset>"
NAME="ntuh_swinv2"
RUN=1

python extract_swinv2_frame_feature.py \
  --dataset "ntucholecvs" \
  --datadir "${DATADIR}/raw" \
  --metadatadir "${DATADIR}/artifact/metadata" \
  --videometapath "${DATADIR}/artifact/metadata/Document03_ annotation summary of NTUcholeCVS videos.csv" \
  --disable_minmaxscale \
  --ckptpath "${ROOTDIR}/model/${NAME}/run_${RUN}/best_ckpt.pt" \
  --num_workers 2 \
  --device "cuda:0" \
  --featdir "${ROOTDIR}/artifact/ntucholecvs/frame_feature/${NAME}/run_${RUN}"
