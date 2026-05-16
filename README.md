# BrainSimSiam

BrainSimSiam is a self-supervised pretraining framework for learning subject-level brain representations from paired graph and 3D image views in fMRI. The repository includes SimSiam-style pretraining, downstream finetuning, and GNNExplainer-based explainability utilities.

## Paper

ArXiv: **[placeholder: add arXiv link here]**

If you use this code, please cite the paper once the citation information is available.

```bibtex
@article{brainsimsiam_placeholder,
  title   = {BrainSimSiam},
  author  = {PLACEHOLDER},
  journal = {arXiv preprint},
  year    = {PLACEHOLDER},
  url     = {PLACEHOLDER_ARXIV_URL}
}
```

## Acknowledgements

Parts of the model structure and self-supervised training logic are adapted from the official SimSiam implementation:

- SimSiam: <https://github.com/facebookresearch/simsiam>

The explainability module is adapted from GNNExplainer / PyTorch Geometric explainability code:

- GNNExplainer paper: <https://arxiv.org/abs/1903.03894>
- PyTorch Geometric: <https://github.com/pyg-team/pytorch_geometric>

Please also cite the original SimSiam and GNNExplainer works when appropriate.

## Repository Structure

```text
.
|-- pretrain_brainsimsiam.py         # Main BrainSimSiam pretraining script
|-- pretrain_brainsimsiam_womask.py  # Pretraining ablation without joint node-image masking
|-- finetune-mlp.py                  # Frozen encoder embedding + MLP downstream evaluation
|-- finetune-end-to-end.py           # End-to-end encoder + predictor finetuning
|-- explain_node_importance.py       # GNNExplainer-based node/edge importance extraction
|-- imports/
|   `-- DualrepData.py               # Graph/image dataset utilities
|-- models/
|   |-- encoder_models.py            # SimSiam and ablation encoders, augmentations
|   `-- gnn_explainer_modified.py    # Modified GNNExplainer implementation
`-- datasets/                        # Expected local dataset/sample split files
```

## Environment

The current local environment used while preparing this README reports:

```text
torch==2.2.2
torch-geometric==2.7.0
torch-sparse==0.6.18
pandas==2.3.3
monai==1.3.2
```

## Data Preparation

The scripts expect local PyTorch serialized data files and sample split files under `datasets/`. The dataset path in the scripts is currently represented by the placeholder string `DIR TO DATA`. Replace it with the directory containing graph/image samples before running experiments.

## Usage

### 1. Pretrain

Run the main BrainSimSiam pretraining script:

```bash
python pretrain_brainsimsiam.py --num_epoch 100 --dim 1024 --node_prob 0.5 --edge_prob 0.5 --drop_prob 0.1 --alpha 1.0 --num_roi 268
```

We also provide the no-mask variation excluding the ROI-image masking augmentation:

```bash
python pretrain_brainsimsiam_womask.py --num_epoch 100 --dim 1024 --node_prob 0.5 --edge_prob 0.5 --drop_prob 0.1 --alpha 1.0 --num_roi 268
```

### 2. Finetune

Option A: train an MLP on frozen pretrained embeddings.

```bash
python finetune-mlp.py --dim 1024 --path weights
```

This script loads `saved_models/weights`, generates train/test embeddings, saves visualizations under `visualization/`, and evaluates downstream classification/regression heads.

Option B: finetune the encoder and prediction head end-to-end.

```bash
python finetune-end-to-end.py --dim 1024 --path weights
```

This script loads `saved_models/weights` and updates the encoder jointly with the downstream predictor.

### 3. Explainability with GNNExplainer

Run node/edge importance extraction with the modified GNNExplainer:

```bash
python explain_node_importance.py --dim 1024 --path weights
```

The script loads `saved_models/weights` and writes explainability artifacts under `explainer_logs/`.