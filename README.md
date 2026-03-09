# Temporal Memory Enhancement for Semantic Segmentation in Surgical Video

Official code repository for the **MIDL 2026 Oral** paper:
**Temporal Memory Enhancement for Semantic Segmentation in Surgical Video**.  
Paper: [OpenReview PDF](https://openreview.net/pdf?id=arGD0IznGt)

## Overview

This repository implements a temporal memory enhancement framework for surgical video semantic segmentation with:

- **Local memory enhancement** via DPP-based diverse and relevant frame selection.
- **Global memory enhancement** via CVAE-MoG phase-context modeling.

Please see the method overview figure in:

- [`framework.pdf`](framework.pdf)

## Main Results (from paper)

On both CHO and ESD datasets, the proposed method improves mIoU / F-score / wDice while keeping real-time speed.

| Method | CHO mIoU | CHO F-score | CHO wDice | ESD mIoU | ESD F-score | ESD wDice | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ours | **86.6** | **47.8** | **92.7** | **86.9** | **76.4** | **90.5** | 66.3 |

For qualitative and analysis figures used in this project, you can also check:

- `model_comparison.png`
- `vis_results/perclass_combined_radars.pdf`
- `CHO_memory.pdf`, `ESD_memory.pdf`

## Repository Structure

- `main.py`: entry point (train + evaluate + save model).
- `runner.py`: training/evaluation workflow.
- `cfgs/`: Hydra configs (model, training, datasets, evaluation).
- `dataset/`: dataset loaders for ESD and CHO (Endoscapes).
- `models/`: network modules (encoder, memory, decoder, losses).
- `evaluation.py`: metrics and evaluator.

## Environment Setup

### Option 1: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate esd_seg
```

### Option 2: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Note: this project depends on `sam2` (see `requirements.txt` editable entry and model config path `sam2/checkpoints/sam2.1_hiera_large.pt`).

## Data Preparation

Two dataset formats are supported.

---

### 1) ESD (`esd_seg_.npy`)

Place your ESD numpy file at project root (or any path you pass in config), for example:

```text
ESD_seg/
  esd_seg_.npy
```

The loader expects a python dict in `.npy`:

- Preferred format (video-wise dict):
  - `{video_id: [(image, mask), (image, mask), ...], ...}`
- Also supported (flattened dict):
  - `{"image": [...], "mask": [...]}`

Expected data types/shapes:

- `image`: RGB frame or clip data (numpy arrays).
- `mask`: label map with integer class ids.

By default, ESD config file is `cfgs/dataset_ESD.yaml` and you can override its `dataloaders.path`.

---

### 2) CHO (Endoscapes)

Create an `endoscapes/` folder under project root (or set `dataloaders.data_root`):

```text
ESD_seg/
  endoscapes/
    train_seg_vids.txt
    val_seg_vids.txt
    test_seg_vids.txt
    semseg/
      <videoId_frameId>.png
    train_seg/
      <videoId_frameId>.jpg or .png
    train/
      <videoId_frameId>.jpg or .png
    val_seg/
      <videoId_frameId>.jpg or .png
    val/
      <videoId_frameId>.jpg or .png
    test_seg/
      <videoId_frameId>.jpg or .png
    test/
      <videoId_frameId>.jpg or .png
```

Filename format must be:

- `<video_id>_<frame_idx>.jpg` / `.png` for images
- `<video_id>_<frame_idx>.png` for masks

Each split txt file contains video IDs (one per line).

## How to Run

All commands are run from repository root.

### Train + evaluate on CHO (default experiment config)

```bash
python main.py --args cfgs --cuda 0
```

### Train + evaluate on ESD (`esd_seg_.npy`)

```bash
python main.py --args cfgs --cuda 0 \
  dataloaders=dataset_ESD \
  dataloaders.path=/absolute/path/to/esd_seg_.npy \
  evaluation.evaluator.num_cls=4
```

### Evaluate from an existing checkpoint (skip training)

```bash
python main.py --args cfgs --cuda 0 \
  training.checkpoint_path=/absolute/path/to/checkpoint.pth
```

### Useful overrides

- Change save directory:

```bash
python main.py --args cfgs --cuda 0 evaluation.save_dir=checkpoints/
```

- Change training epochs:

```bash
python main.py --args cfgs --cuda 0 training.epochs=300
```

## Outputs

- Checkpoints and logs are saved under `evaluation.save_dir` (default: `checkpoints/`).
- Test metrics are appended to:
  - `checkpoints/test_results_CHO.json` or `checkpoints/test_results_ESD.json` (depending on dataset).
- Model weights are saved as:
  - `checkpoints/<DATASET>_<MODEL_NAME>.pth`

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{gao2026temporal,
  title={Temporal Memory Enhancement for Semantic Segmentation in Surgical Video},
  author={Gao, Zheyao and Wu, Qian and Chen, Yueyao and Chen, Cheng and Yip, Hon Chi and Chu, Winnie Chiu Wing and Dou, Qi},
  booktitle={Medical Imaging with Deep Learning (MIDL)},
  year={2026}
}
```

## Acknowledgement

- [SAM2](https://github.com/facebookresearch/sam2)
- [Endoscapes dataset](https://github.com/CAMMA-public/Endoscapes)

