#!/usr/bin/env bash
# Train TFLOP + GAP loss on SynthTabNet.
set -euo pipefail

EXP_NAME="${EXP_NAME:-synthtabnet_gap}"
EXP_VERSION="${EXP_VERSION:-$(date +%Y%m%d_%H%M%S)}"
RESULT_PATH="${RESULT_PATH:-results}"
TOKENIZER="hyunwoongko/asian-bart-ecjk"
DEVICES="${DEVICES:-[0,1]}"

python3 train.py \
  --exp_config  configs/general_exp.yaml \
  --data_config configs/data_synthtabnet.yaml \
  exp_name="${EXP_NAME}" \
  exp_version="${EXP_VERSION}" \
  result_path="${RESULT_PATH}" \
  pretrained_tokenizer_name_or_path="${TOKENIZER}" \
  max_length=1376 \
  bbox_token_cnt=864 \
  train_batch_size=8 \
  val_batch_size=4 \
  num_workers=4 \
  use_OTSL=True \
  use_imgRoiAlign=True \
  use_bbox_HiMulConET=True \
  use_RowWise_contLearning=True \
  use_ColWise_contLearning=True \
  span_coeff_mode=proportional \
  lr=0.00008 \
  max_steps=150000 \
  val_check_interval=0.5 \
  strategy=ddp \
  devices="${DEVICES}" \
  use_adjacent_penalty=True
