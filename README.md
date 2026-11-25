# NeuroHubFormer

NeuroHubFormer learns subject-specific brain hubs from functional connectivity (FC) and uses them to guide two views of the FC matrix:

- **Hub scorer.** Each ROI is rated by its connection profile, the density and the coverage of its learned interaction gates; the three facets are combined into a hub score per ROI.
- **Token view.** A Transformer over the ROI tokens, where the hub scores modulate the tokens and act as an attention prior.
- **Geometric view.** The FC matrix is reweighted by the hub scores through a congruence on the SPD manifold and embedded in the log-Euclidean tangent space.

Both views are concatenated and classified. The model is trained with cross-entropy plus a gate-entropy and a coverage loss on the interaction gates.

## Repository layout

```
main.py                   training and 10-fold cross-validation
model/neurohubformer.py   the model
data/dataset.py           data loading and splits
data/splits/<dataset>/    the 10-fold train/val/test indices used in the paper
run.sh                    the selected configuration for each dataset
```

## Installation

```bash
pip install -r requirements.txt
```

## Data

Each dataset is a single file saved with `torch.save` that holds a dict with

- `signals`: one BOLD signal tensor of shape `[N_rois, T]` per subject,
- `labels`: a 1-D tensor with one class label per subject.

Each subject is represented by the Pearson correlation matrix of its signals. Put the files in a folder (default `./datasets`) under the file names in `DATASET_FILES` in `data/dataset.py`, or pass a file directly with `--data_path`. The paper uses the Schaefer-100 atlas on ABIDE, ADNI, PPMI, HCP-Gender and HCP-Age.

The fold indices in `data/splits/` refer to the subject order of these files. For a new dataset, the splits are created on first use (stratified 10-fold, seed 42; 11% of the non-test subjects of each fold form the validation set).

## Training

```bash
python main.py --data_name abide_schaefer100 --data_root ./datasets \
    --lr 3e-4 --num_layers 2 --dropout 0.2 --interaction_dim 64 --shrinkage 0.02
```

`sh run.sh` runs the selected configuration of every dataset. For each fold, the test metrics are taken at the epoch with the highest validation accuracy; the mean and standard deviation over the folds are printed and appended to `--csv_path`. Use `--checkpoint_dir` to save the best model of each fold.

### Hyperparameters

| Setting | Value |
|---|---|
| Hidden size / heads / FFN width | 128 / 8 / 128 |
| Encoder layers `--num_layers` | {1, 2} |
| Dropout `--dropout` | {0.1, 0.2, 0.3} |
| Interaction width `--interaction_dim` | {32, 64, 128} |
| Shrinkage `--shrinkage` | {0.01, 0.02, 0.05, 0.1, 0.2} |
| Loss weights (entropy / coverage) / target coverage | 0.3 / 2.0 / 0.25 |
| Optimizer | AdamW, weight decay 0 |
| Learning rate `--lr` | {0.5, 1, 3, 5, 7, 10} × 1e-4 |
| Schedule | 10% linear warm-up, cosine decay to 0.1 × peak |
| Epochs / batch size / gradient clipping | 100 / 128 / 5.0 |

Sets are the values searched; all experiments use seed 42.

### Hub scores

`model(x, return_hub_scores=True)` returns the logits and a dict with the per-ROI hub logits, hub scores, profile scores, density and coverage of each subject.
