# MABe Challenge — Mouse Behavior Detection

Kaggle competition pipeline for automated mouse behavior annotation from pose tracking data. Multi-lab dataset with varying tracking setups, body parts, and behavior vocabularies. Annotations are **sparse** — each lab annotated only a subset of behaviors per video. Scoring is interval-level F-beta averaged across laboratories.

This project explores **three architectures**:

1. **CNN + Transformer** — supervised model over the full 37-action vocabulary (`cnn_transformer/main_arc_cnn_transformer/`).
2. **Self + Pair models** — two specialized CNN+Transformer models, one for **self** actions and one for **pair** actions, fused at submission time (`cnn_transformer/self_pair_model/`).
3. **Self-supervised pretraining (PoseBERT)** — masked-autoencoder transformer over single-mouse pose sequences; its embeddings feed into methods 1 and 2 (`pose_bert/`).


## Stage 1 — Preprocessing

### Supervised dataset (per directed pair)

```bash
python preprocessing/build_npy_dataset.py \
    --train_csv      <path/to/train.csv> \
    --tracking_dir   <path/to/train_tracking> \
    --annotation_dir <path/to/train_annotation> \
    --output_dir     <path/to/output_dir>
```

Per directed `(agent, target)` pair:

| File | Shape | Notes |
|---|---|---|
| `{agent}_{target}_features.npy`  | `[T, 176]` float32 | agent ‖ target, each 88-d |
| `{agent}_{target}_labels.npy`    | `[T,  37]` float32 | multi-hot per frame |
| `{agent}_{target}_loss_mask.npy` | `[37]`     float32 | which actions are supervised for this pair |

Plus a global `action_list.npy` and `index.csv` at the dataset root.

### Pretraining dataset (single mouse, SSL)

```bash
python preprocessing/build_pretrain_data.py \
    --train_csv    <path/to/train.csv> \
    --tracking_dir <path/to/train_tracking> \
    --output_dir   <path/to/pose_bert_output> \
    --parts_version v2     # v1 = 11 body parts, v2 = 16 body parts
```

Each mouse produces `[T, n_parts*3]` float32 (x, y, mask interleaved).

---

## Method 3 — PoseBERT (SSL pretraining)

Masked-autoencoder transformer over single-mouse pose sequences. Five variants share the same trainer, selected via `--model`:

| `--model` | Mask mode | Target |
|---|---|---|
| `pos_raw`      | single span | reconstruct (x,y) — MSE |
| `pos_bins`     | single span | classify binned position — CE |
| `vel_bins`     | single span | classify binned velocity — CE |
| `pos_vel_bins` | two disjoint spans | joint binned position + velocity — CE |
| `forecast`     | none (autoregressive) | predict next-step poses |

Train:

```bash
python -m pose_bert.model.train <path/to/pose_bert_output> \
    --model        pos_vel_bins \
    --output_dir   <path/to/pose_bert_ckpt> \
    --num_epochs   50 --batch_size 64 --lr 1e-4 \
    --d_model      256 --num_layers 6 --nhead 8 \
    --window_size  128 --stride 64 \
    --metadata_csv <path/to/train.csv> \
    --augment --run_name pos_vel_bins_v2
```

Extract embeddings for all mice (train + test):

```bash
python -m pose_bert.scripts.extract_embeddings \
    --checkpoint <path/to/pose_bert_ckpt> \
    --source     <path/to/pose_bert_output> \
    --output     <path/to/embeddings> \
    --train_csv  <path/to/train.csv> \
    --stitch     center
```

Writes `{lab}/{video}/{mouse_id}.npy` of shape `[T, d_model]`.

---

## Methods 1 & 2 — Supervised CNN + Transformer

Both methods share the same per-frame architecture:

```
Input [B, T, F]
  → Linear projection (F → d_model)
  → TemporalCNN (2× Conv1d, kernel=3, residual)
  → + Learned positional encoding
  → TransformerEncoder (pre-norm GELU + relative-position attention bias)
  → Linear head (d_model → 37)
Output [B, T, 37]   (raw logits → sigmoid)
```

Loss: `BCEWithLogitsLoss`, masked by `loss_mask` (unsupervised actions zeroed) and `padding_mask`.

Both accept `--features_mode {raw, embed, concat}`:

| Mode | `F` | What the model sees |
|---|---|---|
| `raw`    | 176             | hand-crafted features only |
| `embed`  | `2·d_model`     | PoseBERT embeddings only |
| `concat` | 176 + 2·d_model | both, concatenated |

`embed` and `concat` require `--embeddings_dir`.

### Method 1 — main arc (single model, 37 actions)

```bash
python cnn_transformer/main_arc_cnn_transformer/model/model.py \
    <path/to/processed> <path/to/checkpoint_dir> \
    --num_epochs 50 --batch_size 32 --lr 1e-4 \
    --window_size 64 --stride 32 \
    --features_mode concat \
    --embeddings_dir <path/to/embeddings> \
    --train_csv <path/to/train.csv> \
    --run_name
```

### Method 2 — self + pair models (two specialized models)

```bash
# Pair-actions model
python cnn_transformer/self_pair_model/model/model.py \
    <path/to/processed> <path/to/pair_ckpt_dir> \
    --train_kind pair --features_mode raw --num_epochs 50

# Self-actions model
python cnn_transformer/self_pair_model/model/model.py \
    <path/to/processed> <path/to/self_ckpt_dir> \
    --train_kind self --features_mode raw --num_epochs 50
```

### Common flags

- `--val_split / --seed` — video-level split (no window leakage)
- `--beta` — F-beta β (default 1.0)
- `--aug_*` / `--no_augment` — flip, scale jitter, rotation, per-part dropout, time crop
- `--meta_unk_p` — UNK rate for metadata embeddings
- Device auto-selects **CUDA > MPS > CPU**.

---

## Stage 5 — Threshold tuning

```bash
python cnn_transformer/main_arc_cnn_transformer/scripts/tune_thresholds.py \
    --checkpoint  <path/to/best_model.pt> \
    --data_dir    <path/to/processed> \
    --train_csv   <path/to/train.csv> \
    --window_size 64 --stride 32
```

Produces a JSON of tuned per-(lab, action) thresholds. The `ensemble_eval` / `ensemble_eval_mixed` scripts combine multiple checkpoints and re-tune jointly.

---

## Stage 6 — Inference / Submission

**Method 1 — single model:**

```bash
python cnn_transformer/main_arc_cnn_transformer/model/inference.py \
    --model_path <path/to/best_model.pt> \
    --output     submission.csv
```

**Method 1 — ensemble:**

```bash
python cnn_transformer/main_arc_cnn_transformer/model/inference_ensemble.py \
    --model_paths <path/to/ckpt1.pt> <path/to/ckpt2.pt> ... \
    --output      submission.csv
```

**Method 2 — self + pair fused:**

```bash
python cnn_transformer/self_pair_model/model/inference.py \
    --pair_model_path <path/to/pair_best.pt> \
    --self_model_path <path/to/self_best.pt> \
    --output          submission.csv
```

Each inference script: parquet → preprocess on the fly → sliding-window forward pass → average overlapping logits → apply tuned thresholds → convert to `(start_frame, stop_frame)` intervals → write `submission.csv`.

Kaggle environment runs **inference only** (9h runtime, 20 GB disk, ~13 GB RAM).

---

## Feature engineering — 176 dims per frame

Each mouse contributes 88-d per frame; a directed pair concatenates `[agent ‖ target]` → 176-d.

| Block | Columns | Count |
|---|---|---|
| Canonical positions          | `x_{part}`, `y_{part}`       | 22 |
| Position observation masks   | `m_x_{part}`, `m_y_{part}`   | 22 |
| Velocities (×fps)            | `vx_{part}`, `vy_{part}`     | 22 |
| Velocity masks               | `m_vx_{part}`, `m_vy_{part}` | 22 |

Body parts (11): `body_center`, `ear_left`, `ear_right`, `hip_left`, `hip_right`, `lateral_left`, `lateral_right`, `neck`, `nose`, `tail_base`, `tail_tip`.

Coordinates are **agent-centric**: per frame, origin = midpoint of agent's ears, rotated so the agent's heading vector aligns with the negative-y axis. Both agent and target share the same per-frame transform. Missing keypoints are interpolated for geometry; masks (computed *before* fill) record what was actually observed.

See [`preprocessing/preprocess_features.py`](preprocessing/preprocess_features.py).

## Labels, metadata, scoring

- **Action list (`A=37`)** — every action in any lab's `behaviors_labeled` whitelist.
- **Labels** `[T, 37]` — multi-hot per frame per directed pair.
- **Loss mask** `[37]` — per-pair: 1.0 for supervised actions, else 0.0. BCE multiplied by this mask so unsupervised slots don't contribute to the gradient.
- **Metadata embeddings** — `lab`, `strain (×2)`, `sex (×2)` per pair from `train.csv`, used by both PoseBERT and the supervised models. UNK'd with `--meta_unk_p`.
- **Scoring** — interval-level F-beta averaged across labs (see [`cnn_transformer/F_Beta.py`](cnn_transformer/F_Beta.py)). Training tracks **frame-level** F-beta over supervised slots only.

## Training environments

| Environment | Purpose |
|---|---|
| Mac M3 Pro (MPS) | Dev, quick iterations |
| GCP T4 / A100    | Long pretraining + supervised runs |
| Google Colab     | Alternative training |
| Kaggle           | Inference + submission only |

## References

See [`cnn_transformer/main_arc_cnn_transformer/links.md`](cnn_transformer/main_arc_cnn_transformer/links.md) for the Kaggle competition page and data-constraints notebook.
