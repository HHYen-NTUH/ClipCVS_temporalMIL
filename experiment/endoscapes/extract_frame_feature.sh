#!/bin/bash

set -e

ROOTDIR="<repository directory>"
DATADIR="<directory of Endoscapes dataset>"

python extract_swinv2_frame_feature.py \
  --dataset "endoscapes" \
  --datadir "${DATADIR}" \
  --disable_minmaxscale \
  --swinv2_ckptpath "${ROOTDIR}/pretrained/swincvs/Swin_backbone_no_augm_sd4_bestMAP.pt" \
  --num_workers 2 \
  --device "cuda:0" \
  --featdir "${ROOTDIR}/artifact/endoscapes/frame_feature/pretrained_swincvs_frozen_backbone"
