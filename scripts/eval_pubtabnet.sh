#!/usr/bin/env bash
# Run inference + TEDS evaluation for one checkpoint on the PubTabNet test set.
#
# Usage:
#   DATA_ROOT=/path/to/TFLOP-dataset \
#   scripts/eval_pubtabnet.sh <exp_savepath> <epoch_step_checkpoint>
#
# <exp_savepath> is the run directory containing config.yaml, e.g.
#   results/pubtabnet_gap/20260101_120000
# <epoch_step_checkpoint> is a checkpoint subdirectory, e.g. epoch_30_step_117425
set -euo pipefail

EXP_SAVEPATH="${1:?usage: eval_pubtabnet.sh <exp_savepath> <epoch_step_checkpoint>}"
CKPT="${2:?usage: eval_pubtabnet.sh <exp_savepath> <epoch_step_checkpoint>}"

DATA_ROOT="${DATA_ROOT:-./data/TFLOP-dataset}"
AUX_JSON="${AUX_JSON:-${DATA_ROOT}/meta_data/final_eval_v2.json}"
AUX_IMG="${AUX_IMG:-${DATA_ROOT}/images/test}"
AUX_REC_PKL="${AUX_REC_PKL:-${DATA_ROOT}/pse_results/test/end2end_results.pkl}"
BATCH_SIZE="${BATCH_SIZE:-12}"

CKPT_DIR="${EXP_SAVEPATH}/${CKPT}"

python3 test.py \
  --tokenizer_name_or_path "${CKPT_DIR}" \
  --model_name_or_path     "${CKPT_DIR}" \
  --exp_config_path        "${EXP_SAVEPATH}/config.yaml" \
  --model_config_path      "${CKPT_DIR}/config.json" \
  --aux_json_path          "${AUX_JSON}" \
  --aux_img_path           "${AUX_IMG}" \
  --aux_rec_pkl_path       "${AUX_REC_PKL}" \
  --batch_size             "${BATCH_SIZE}" \
  --save_dir               "${CKPT_DIR}" \
  --current_bin 0 --num_bins 1

python3 evaluate_ted.py \
  --model_inference_pathdir "${CKPT_DIR}" \
  --output_savepath         "${CKPT_DIR}"

python3 scripts/report_metrics.py "${CKPT_DIR}"
