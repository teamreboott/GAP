# GAP: Geometry-Aware Pointer Loss for Spatial Locality

Official implementation of **"Rethinking the Pointer Loss in Table Structure
Recognition: Geometry-Aware Pointer Loss for Spatial Locality"**.

[![Paper](https://img.shields.io/badge/arXiv-2606.18721-b31b1b.svg)](https://arxiv.org/abs/2606.18721)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](./LICENSE)

## Overview

Pointer networks for Table Structure Recognition (TSR) align predicted structure
tags to detected text regions with a softmax pointer loss that treats every
negative candidate identically. Our analysis shows this is a poor match for how
these models actually fail: **79.6% of pointer errors land on cells within
Manhattan distance d ≤ 2** of the ground truth, yet those near-miss candidates
receive only 26.7% of the gradient signal.

**GAP loss** reweights the pointer cross-entropy by spatial proximity, so
gradients concentrate where confusions actually occur. It is a loss-level change
only — the architecture is untouched and inference cost is unchanged.

## Method

Negative candidates are weighted by their Manhattan distance `d` to the
ground-truth cell in the table grid:

```
w(d) = max(alpha / 2**d, 0.5),    d >= 1,    alpha = 8
```

which yields weights of **4, 2, 1, 0.5** for `d = 1, 2, 3, 4`, and `0.5` for
`d >= 5`. The negative weights are then mass-normalised to sum to `kappa`, so
the total negative mass stays fixed and only its *distribution* changes.

> **Erratum.** Eq. (7) of the arXiv paper prints this as `alpha / 2**(d-1)`; the
> exponent there is a typo. The weights quoted in the same paragraph ("4, 2, 1
> for distances 1, 2, 3") and Fig. 4 both correspond to `alpha / 2**d`, which is
> what this implementation and the released checkpoints use.

Implementation: [`tflop/model/decoder/mbart_decoder_weighted.py`](tflop/model/decoder/mbart_decoder_weighted.py)

## Results

PubTabNet test set and SynthTabNet validation set:

| Method | PubTabNet TEDS-S | PubTabNet TEDS | SynthTabNet TEDS-S | SynthTabNet TEDS |
|---|---|---|---|---|
| TFLOP | 98.27 | 96.43 | 99.53 | 99.25 |
| **TFLOP + GAP** | **98.28** | **96.49** | **99.69** | **99.53** |

Position Accuracy (PA), a strict cell-level correctness metric:

| Method | PubTabNet | SynthTabNet |
|---|---|---|
| TFLOP | 91.80 | 98.75 |
| **+ GAP Loss** | **92.35** | **99.31** |

## Repository layout

```
GAP/
├── train.py                  # training entrypoint
├── test.py                   # inference over a test split
├── evaluate_ted.py           # TEDS / TEDS-Struct scoring
├── configs/
│   ├── general_exp.yaml      # model + GAP loss settings
│   ├── data_pubtabnet.yaml   # PubTabNet paths
│   └── data_synthtabnet.yaml # SynthTabNet paths
├── scripts/
│   ├── train_pubtabnet.sh
│   ├── train_synthtabnet.sh
│   ├── eval_pubtabnet.sh
│   └── report_metrics.py     # TEDS / TEDS-S / PA summary
└── tflop/
    ├── model/
    │   ├── decoder/
    │   │   ├── mbart_decoder.py           # pointer loss path
    │   │   └── mbart_decoder_weighted.py  # GAP loss
    │   ├── model/                         # TFLOP model + config
    │   └── visual_encoder/                # Swin backbone
    ├── datamodule/                        # datasets & preprocessing
    ├── lightning_module/                  # training loop
    └── evaluator.py                       # TEDS implementation
```

## Installation

```bash
conda create -n gap python=3.9
conda activate gap

pip install torch==2.0.1 torchmetrics==1.6.0 torchvision==0.15.2
pip install -r requirements.txt
```

## Data

This repository follows the TFLOP data layout. Download the dataset from
[upstage/TFLOP-dataset](https://huggingface.co/datasets/upstage/TFLOP-dataset):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
git lfs install
git clone https://huggingface.co/datasets/upstage/TFLOP-dataset
```

Then point `image_path` and `meta_data_path` in
`configs/data_pubtabnet.yaml` at the extracted directories.

PubTabNet's test split ships without ground-truth cell bounding boxes, so
evaluation uses the pre-extracted OCR boxes released with TFLOP.

## Training

```bash
# TFLOP + GAP on PubTabNet (2 GPUs by default)
DEVICES="[0,1]" scripts/train_pubtabnet.sh

# SynthTabNet
DEVICES="[0,1]" scripts/train_synthtabnet.sh
```

Or call `train.py` directly — it merges an experiment config, a data config, and
any `key=value` overrides from the command line:

```bash
python3 train.py \
  --exp_config  configs/general_exp.yaml \
  --data_config configs/data_pubtabnet.yaml \
  exp_name=pubtabnet_gap result_path=results \
  use_adjacent_penalty=True devices="[0,1]"
```

GAP is controlled entirely from the config:

```yaml
use_adjacent_penalty: True    # False reverts to the standard pointer CE
adjacent_penalty_config:
  alpha: 8.0                  # base weight scale
  min_weight: 0.5             # floor for distant candidates
  kappa: 1.0                  # total negative mass after normalisation
  temperature: 0.1
  use_spatial_weights: True
  use_mass_normalization: True
  use_logitnorm: True
```

Setting `use_adjacent_penalty: False` reproduces the TFLOP baseline, so an
ablation is a one-line change.

Paper configuration: input 768×768, batch size 64, `lambda_1 = lambda_2 = 1.0`,
`lambda_3 = 0.5`, trained on 2×H200.

## Inference & Evaluation

```bash
DATA_ROOT=./data/TFLOP-dataset \
scripts/eval_pubtabnet.sh results/pubtabnet_gap/<version> <epoch_step_checkpoint>
```

This runs inference (`test.py`), computes tree-edit-distance scores
(`evaluate_ted.py`), and prints the summary:

```
checkpoint : results/pubtabnet_gap/.../epoch_30_step_117425
TEDS       : 96.49
TEDS-Struct: 98.28
PA         : 92.35
```

Reported metrics are TEDS, TEDS-Struct, and Position Accuracy (PA). PA counts a
cell as correct only if its pointer maps to exactly the right text region — it
exposes systematic misalignments that TEDS masks through partial credit.

## Citation

```bibtex
@article{choi2026gap,
  title={Rethinking the Pointer Loss in Table Structure Recognition:
         Geometry-Aware Pointer Loss for Spatial Locality},
  author={Choi, Hong-Jun and Lee, Jongho and Kim, Jaeyoung},
  journal={arXiv preprint arXiv:2606.18721},
  year={2026}
}
```

## Acknowledgement

This work builds directly on **TFLOP: Table Structure Recognition Framework with
Layout Pointer Mechanism** (IJCAI 2024) by Minsoo Khang and Teakgyu Hong. Our
codebase is derived from their official implementation, and GAP replaces only
the pointer loss while leaving their architecture and training pipeline intact.
We thank the authors for releasing their code, dataset, and pretrained weights.

- Paper: https://www.ijcai.org/proceedings/2024/0105.pdf
- Code: https://github.com/UpstageAI/TFLOP

```bibtex
@inproceedings{khang2024tflop,
  title={TFLOP: Table Structure Recognition Framework with Layout Pointer Mechanism},
  author={Khang, Minsoo and Hong, Teakgyu},
  booktitle={Proceedings of the Thirty-Third International Joint Conference
             on Artificial Intelligence (IJCAI)},
  pages={947--955},
  year={2024}
}
```

## License

Released under [CC BY-NC 4.0](./LICENSE), inherited from the upstream TFLOP
repository. **Non-commercial use only.** See [NOTICE](./NOTICE) for attribution
details and the list of modifications.
