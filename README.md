# Clip-level Temporal MIL for CVS Classification

This repository contains experiments in "Revisiting Photo Versus Video Assessment of the Critical View of Safety in Laparoscopic Cholecystectomy: A Deep Learning Study".

This document provides a step-by-step to guide to perform the experiments. Experiment results will be stored in the `analysis` directory. If you wish to use our pre-trained checkpoints, modify the bash scripts to use the `pretrained` directory instead of the `model` directory.

## Preparation

1. Create conda environment and activate it:

```bash
conda env create -f environment.yml
conda activate cvs-clip-temporal-mil
```

2. Download the  [Endoscapes dataset](https://github.com/CAMMA-public/Endoscapes).
3. Download the [SwinCVS checkpoints](https://github.com/franeknowak/SwinCVS) under `pretrained/swincvs/`.
4. If you wish not to train your own model, you can download our [pre-trained checkpoints](https://fethb3053wy-my.sharepoint.com/:f:/g/personal/chiahungyang_stmedical_tw/IgAFE6WK-gfXRpiFGav3XxY6AdZvoZ-vv8GAOZyeEy3dcZk?e=wbfXTI) into `pretrained`.

## Endoscapes dataset

### Extract frame features

Run

```bash
bash experiment/endoscapes/extract_frame_feature.sh
```

### Replicating SwinCVS LSTM temporal branch

Run

```bash
bash experiment/endoscapes/lstm.sh
```

### Temporal MILs with various clip lengths

Run

```bash
bash experiment/endoscapes/temporal_mils.sh
```

### Aggregating baseline prediction

Run

```bash
bash experiment/endoscapes/aggregate_baseline.sh
```

## NTUH dataset

### Fine-tuning SwinV2

Run

```bash
bash experiment/ntuh/train_swinv2.sh
```

### Extract frame features

Run

```bash
bash experiment/ntuh/extract_frame_feature.sh
```

### Replicating SwinCVS LSTM temporal branch

Run

```bash
bash experiment/ntuh/lstm.sh
```

### Temporal MILs with various clip lengths

Run

```bash
bash experiment/ntuh/temporal_mils.sh
```

### Aggregating baseline prediction

Run

```bash
bash experiment/ntuh/aggregate_baseline.sh
```
