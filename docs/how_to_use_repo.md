# How to Use This Repo

## Setup

- Ensure the ArtBench-10 Kaggle data is in `data/`:
  - `data/ArtBench-10.csv`
  - `data/artbench-10-python/artbench-10-batches-py/`

## Configs

- Training uses YAML configs in `configs/`.
- Example: `configs/vae_config.yml`
- Key fields include: `model_type`, `device`, `optimizer`, `batch_size`, `epochs`, `kaggle_root`.

## Train

```bash
python src/train.py configs/dcgan_config.yml --checkpoint outputs/run_dcgan_20260317/model.pt
```
**Note:** ```--checkpoint``` parameter is optional

- Outputs are written to `outputs/run_<model>_<timestamp>/`.

## Evaluate

```bash
python src/evaluate.py dcgan
```

- Evaluation loads the latest VAE checkpoint from `outputs/run_vae_*`.
- To change sample counts, edit `src/evaluate.py` defaults.

## Add a New Model (!! Modular !! 👈🤣)

1) Implement the model in `src/models/`.
2) Add a `model_type` entry in the YAML config.
3) Extend the model selection block in `src/train.py` and raise `NotImplementedError` until implemented.
4) Add a sampler in `src/evaluate.py` for that model.

## Weights & Biases (W&B) for Team Members

This repo has optional W&B logging in both training and evaluation.

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Configure local credentials

Create or update `.env` in the repo root with:

```env
WANDB_API_KEY=your_wandb_api_key
WANDB_ENTITY=nunombatista-university-of-coimbra
WANDB_PROJECT=ArtBench-AEGANDIFF
```

Notes:
- `.env` is git-ignored, so each team member should keep their own key locally.
- Do not commit API keys to the repository.

### 3) Enable/disable W&B in config

In your training config (for example `configs/diffusion_config.yml`):

```yml
wandb:
  enabled: true
  entity: "nunombatista-university-of-coimbra"
  project: "ArtBench-AEGANDIFF"
  tags: ["diffusion", "artbench"]
```

Set `enabled: false` to run without W&B.

### 4) Run training with live logging

```bash
python src/train.py configs/diffusion_config.yml
```

This logs epoch metrics live and uploads run artifacts at the end (checkpoints, config, history, metrics, and sample image).

### 5) Run evaluation with W&B logging

```bash
python src/evaluate.py diffusion
```

This logs evaluation metrics (`fid`, `kid_mean`, `kid_std`) to W&B.
