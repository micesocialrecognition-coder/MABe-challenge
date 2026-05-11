"""Ensemble + TTA Kaggle submission for MABe (multi-checkpoint averaging).

Usage: python inference_ensemble.py --model_paths ckpt1.pt ckpt2.pt [--tta_flip]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PROJECT_ROOT.parent.parent / "preprocessing"))

from model import MABeTransformer, apply_augmentations, _rotate_block  # noqa: E402
from metadata import (  # noqa: E402
    N_META_SLOTS,
    build_pair_lookup,
    build_vocabs,
    resolve_meta_ids,
    slot_to_table_index,
    vocab_sizes_per_table,
)
from preprocess_features import (  # noqa: E402
    ALL_PARTS,
    DROP_PARTS,
    process_directed_pair_agent_centric,
)
from preprocess_data import build_action_universe  # noqa: E402


DEFAULT_MODEL_CONFIG = dict(
    input_dim=176,
    num_classes=37,
    d_model=256,
    nhead=4,
    num_layers=3,
    dim_feedforward=1024,
    dropout=0.2,
    window_size=64,
)
WINDOW_SIZE = 64
STRIDE = 32


def find_kaggle_competition_root() -> Path | None:
    """Locate the MABe competition root under /kaggle/input/; None off-Kaggle."""
    ki = Path("/kaggle/input")
    if not ki.is_dir():
        return None

    def looks_like_root(p: Path) -> bool:
        return (p / "test.csv").is_file() and (p / "test_tracking").is_dir()

    candidates: List[Path] = []
    for child in sorted(ki.iterdir()):
        if not child.is_dir():
            continue
        if looks_like_root(child):
            candidates.append(child)
            continue
        try:
            for grand in sorted(child.iterdir()):
                if grand.is_dir() and looks_like_root(grand):
                    candidates.append(grand)
        except (PermissionError, OSError):
            pass

    if candidates:
        for c in candidates:
            if c.name.lower() == "mabe-mouse-behavior-detection":
                return c
        return candidates[0]
    return None


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _detect_meta_config_from_state(
    state: Dict,
) -> Tuple[List[int], List[int]]:
    """Return (vocab_sizes, slot_to_table) matching meta_embeddings.* in state."""
    sizes: List[int] = []
    i = 0
    while True:
        key = f"meta_embeddings.{i}.weight"
        if key not in state:
            break
        sizes.append(int(state[key].shape[0]))
        i += 1
    if not sizes:
        return [], []
    slot_to_table = slot_to_table_index()
    return sizes, slot_to_table


def _detect_window_size_from_state(state: Dict, num_heads: int) -> int | None:
    """Recover training-time window_size from rel_pos_bias rows (= 2*window_size - 1)."""
    for k, v in state.items():
        if k.endswith("attention.rel_pos_bias.weight"):
            rows = int(v.shape[0])
            if rows >= 1 and rows % 2 == 1:
                return (rows + 1) // 2
    return None


def load_model(
    model_path: str,
    num_classes: int,
    device: torch.device,
    window_size: int | None = None,
    input_dim: int | None = None,
) -> MABeTransformer:
    cfg = dict(DEFAULT_MODEL_CONFIG)
    cfg["num_classes"] = num_classes
    if input_dim is not None:
        cfg["input_dim"] = input_dim

    state = torch.load(model_path, map_location=device, weights_only=True)

    # Unwrap common checkpoint wrappers; otherwise load_state_dict(strict=False)
    # silently loads zero parameters and the model produces garbage probs.
    if isinstance(state, dict):
        for wrapper_key in ("model", "state_dict", "model_state_dict", "module"):
            if wrapper_key in state and isinstance(state[wrapper_key], dict):
                inner = state[wrapper_key]
                if any(isinstance(v, torch.Tensor) for v in inner.values()):
                    print(f"[load_model] unwrapping checkpoint key '{wrapper_key}'.")
                    state = inner
                    break
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {
            k[len("module."):] if k.startswith("module.") else k: v
            for k, v in state.items()
        }
    if isinstance(state, dict) and any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {
            k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
            for k, v in state.items()
        }

    # Configure metadata layers BEFORE constructing the model so embedding tables match.
    meta_vocab_sizes, meta_slot_to_table = _detect_meta_config_from_state(state)
    cfg["meta_vocab_sizes"] = meta_vocab_sizes or None
    cfg["meta_slot_to_table"] = meta_slot_to_table or None

    detected_ws = _detect_window_size_from_state(state, cfg["nhead"])
    if detected_ws is not None:
        if window_size is not None and window_size != detected_ws:
            print(f"[load_model] --window_size={window_size} overridden by "
                  f"checkpoint's trained window_size={detected_ws}.")
        cfg["window_size"] = detected_ws
    elif window_size is not None:
        cfg["window_size"] = window_size
    print(f"[load_model] model window_size={cfg['window_size']}")
    if meta_vocab_sizes:
        print(f"[load_model] checkpoint has metadata layers: "
              f"vocab_sizes={meta_vocab_sizes}, slot_to_table={meta_slot_to_table}")
    else:
        print("[load_model] checkpoint has no metadata layers — running "
              "without metadata conditioning.")
    model = MABeTransformer(**cfg).to(device)

    has_temporal_cnn = any(k.startswith("temporal_cnn.") for k in state.keys())
    if not has_temporal_cnn:
        print("[load_model] checkpoint has no temporal_cnn weights — replacing "
              "temporal_cnn with Identity to match the training-time architecture.")
        model.temporal_cnn = torch.nn.Identity()

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load_model] missing={missing}\n[load_model] unexpected={unexpected}")

    expected_params = {n for n, _ in model.named_parameters()}
    loaded = expected_params - set(missing)
    if len(loaded) < 0.5 * len(expected_params):
        raise SystemExit(
            f"[load_model] only {len(loaded)}/{len(expected_params)} model "
            "parameters were loaded from the checkpoint. The checkpoint "
            "structure does not match the model."
        )

    model.eval()
    return model


@torch.no_grad()
def predict_video_pair(
    model: MABeTransformer,
    features: np.ndarray,
    device: torch.device,
    window_size: int,
    stride: int,
    batch_size: int,
    meta_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Return per-frame action probabilities [T, num_classes]; overlapping windows averaged."""
    T, F = features.shape
    num_classes = DEFAULT_MODEL_CONFIG["num_classes"]
    if T == 0:
        return np.zeros((0, num_classes), dtype=np.float32)

    # Ensure the final window covers the tail.
    starts: List[int] = list(range(0, max(T - window_size, 0) + 1, stride))
    if not starts:
        starts = [0]
    if starts[-1] + window_size < T:
        starts.append(max(T - window_size, 0))

    prob_sum = np.zeros((T, num_classes), dtype=np.float32)
    prob_cnt = np.zeros((T, 1), dtype=np.float32)

    for chunk_start in range(0, len(starts), batch_size):
        chunk = starts[chunk_start: chunk_start + batch_size]

        x = np.zeros((len(chunk), window_size, F), dtype=np.float32)
        pad = np.ones((len(chunk), window_size), dtype=bool)
        ends = []
        for i, s in enumerate(chunk):
            e = min(s + window_size, T)
            x[i, : e - s] = features[s:e]
            pad[i, : e - s] = False
            ends.append(e)

        x_t = torch.from_numpy(x).to(device)
        pad_t = torch.from_numpy(pad).to(device)
        meta_t = None
        if meta_ids is not None:
            # Same metadata for every window of this pair → tile across batch.
            meta_t = torch.from_numpy(
                np.broadcast_to(meta_ids, (len(chunk), N_META_SLOTS)).copy()
            ).to(device)
        logits = model(x_t, pad_t, meta_t)
        probs = torch.sigmoid(logits).cpu().numpy()

        for i, s in enumerate(chunk):
            e = ends[i]
            valid = e - s
            prob_sum[s:e] += probs[i, :valid]
            prob_cnt[s:e] += 1.0

    prob_cnt[prob_cnt == 0.0] = 1.0
    return prob_sum / prob_cnt


def smooth_probs(probs: np.ndarray, kind: str = "gauss",
                 k: float = 5.0) -> np.ndarray:
    """Smooth probs along time. kind ∈ {none, box (window=k), gauss (sigma=k)}."""
    if probs.shape[0] == 0 or kind == "none" or k is None or k <= 0:
        return probs
    if kind == "gauss":
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(probs, sigma=float(k), axis=0,
                                 mode="nearest").astype(probs.dtype, copy=False)
    if kind == "box":
        window = int(k)
        if window <= 1:
            return probs
        if window % 2 == 0:
            window += 1
        pad = window // 2
        padded = np.pad(probs, ((pad, pad), (0, 0)), mode="edge")
        cs = np.cumsum(padded, axis=0, dtype=np.float64)
        cs = np.concatenate(
            [np.zeros((1, probs.shape[1]), dtype=np.float64), cs], axis=0
        )
        summed = cs[window:] - cs[:-window]
        return (summed / window).astype(probs.dtype)
    raise ValueError(f"unknown smoothing kind: {kind!r}")


def tta_transform(features: np.ndarray, flip: bool = False,
                  angle_deg: float = 0.0) -> np.ndarray:
    """Deterministic geometric TTA on raw 176-d [agent|target] features."""
    if not flip and angle_deg == 0.0:
        return features
    out = features
    if flip:
        out = apply_augmentations(
            out, np.random.default_rng(0),
            flip_p=1.0, scale_jitter=0.0, rot_prob=0.0, rot_max_deg=0.0,
            part_dropout_prob=0.0, part_hide_prob=0.0,
        )
    if angle_deg != 0.0:
        cos_t = float(np.cos(np.deg2rad(angle_deg)))
        sin_t = float(np.sin(np.deg2rad(angle_deg)))
        agent = _rotate_block(out[:, :88], cos_t, sin_t)
        target = _rotate_block(out[:, 88:], cos_t, sin_t)
        out = np.concatenate([agent, target], axis=1).astype(np.float32, copy=False)
    return out


def close_mask(mask: np.ndarray, gap_tol: int) -> np.ndarray:
    """Fill False gaps of length <= gap_tol bracketed by True frames."""
    if gap_tol <= 0 or mask.size == 0:
        return mask

    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    diff = np.diff(padded)
    true_starts = np.where(diff == 1)[0]
    true_ends = np.where(diff == -1)[0]
    if len(true_ends) < 2:
        return mask
    out = mask.copy()
    gap_starts = true_ends[:-1]
    gap_stops = true_starts[1:]
    short = (gap_stops - gap_starts) <= gap_tol
    for s, e in zip(gap_starts[short], gap_stops[short]):
        out[s:e] = True
    return out


def runs_from_mask(
    mask: np.ndarray,
    vf: np.ndarray,
    min_len: int,
) -> List[Tuple[int, int, int, int]]:
    """Return (start_frame, stop_frame_excl, run_start_idx, run_stop_idx_excl)."""
    if mask.size == 0:
        return []
    intervals: List[Tuple[int, int, int, int]] = []
    diff = np.diff(mask.astype(np.int8))
    starts_idx = list(np.where(diff == 1)[0] + 1)
    ends_idx = list(np.where(diff == -1)[0] + 1)
    if mask[0]:
        starts_idx.insert(0, 0)
    if mask[-1]:
        ends_idx.append(len(mask))
    for s_i, e_i in zip(starts_idx, ends_idx):
        if e_i - s_i < min_len:
            continue
        start_frame = int(vf[s_i])
        stop_frame = int(vf[e_i - 1]) + 1
        intervals.append((start_frame, stop_frame, int(s_i), int(e_i)))
    return intervals


def enforce_pair_no_overlap(rows: List[Dict]) -> List[Dict]:
    """Enforce: no two cross-action intervals overlap per (video, agent, target).

    Conflicts resolved by mean-probability `confidence`; lower-confidence side trimmed.
    """
    if not rows:
        return rows

    grouped: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        key = (r["video_id"], r["agent_id"], r["target_id"])
        grouped[key].append(r)

    resolved: List[Dict] = []
    dropped = 0
    trimmed = 0

    for _key, group in grouped.items():
        # Tie-break on confidence so the higher-confidence interval "owns" the start.
        group.sort(key=lambda r: (int(r["start_frame"]), -float(r["confidence"])))
        accepted: List[Dict] = []
        for r in group:
            s = int(r["start_frame"])
            e = int(r["stop_frame"])
            if e <= s:
                dropped += 1
                continue
            keep = True
            for a in accepted:
                a_s = int(a["start_frame"])
                a_e = int(a["stop_frame"])
                if e <= a_s or s >= a_e:
                    continue
                if r["action"] == a["action"]:
                    # Same action: merge with confidence-weighted mean.
                    new_s = min(a_s, s)
                    new_e = max(a_e, e)
                    w_a = a_e - a_s
                    w_r = e - s
                    a["confidence"] = (
                        a["confidence"] * w_a + r["confidence"] * w_r
                    ) / max(w_a + w_r, 1)
                    a["start_frame"] = new_s
                    a["stop_frame"] = new_e
                    keep = False
                    trimmed += 1
                    break
                if r["confidence"] >= a["confidence"]:
                    # New row wins — shrink the accepted interval.
                    if a_s < s and a_e > e:
                        right_half = dict(a)
                        right_half["start_frame"] = e
                        right_half["stop_frame"] = a_e
                        a["stop_frame"] = s
                        accepted.append(right_half)
                        trimmed += 1
                    elif a_s < s:
                        a["stop_frame"] = s
                        trimmed += 1
                    elif a_e > e:
                        a["start_frame"] = e
                        trimmed += 1
                    else:
                        a["start_frame"] = a["stop_frame"] = -1
                        dropped += 1
                else:
                    # Accepted wins — shrink (or drop) the new row.
                    if s < a_s and e > a_e:
                        right_piece = dict(r)
                        right_piece["start_frame"] = a_e
                        right_piece["stop_frame"] = e
                        accepted.append(right_piece)
                        r["stop_frame"] = a_s
                        e = a_s
                        trimmed += 1
                    elif s < a_s:
                        r["stop_frame"] = a_s
                        e = a_s
                        trimmed += 1
                    elif e > a_e:
                        r["start_frame"] = a_e
                        s = a_e
                        trimmed += 1
                    else:
                        keep = False
                        dropped += 1
                        break
            if keep and int(r["stop_frame"]) > int(r["start_frame"]):
                accepted.append(r)
        for a in accepted:
            if int(a["start_frame"]) >= 0 and int(a["stop_frame"]) > int(a["start_frame"]):
                resolved.append(a)

    if trimmed or dropped:
        print(f"[enforce_pair_no_overlap] trimmed/merged={trimmed}, dropped={dropped}, "
              f"final rows={len(resolved)} (from {len(rows)})")
    return resolved


def assert_no_pair_overlap(df: pd.DataFrame) -> None:
    """Raise if any cross-action overlap remains in df."""
    if len(df) == 0:
        return
    grouped: Dict[Tuple, List[Tuple]] = defaultdict(list)
    for r in df.itertuples(index=False):
        grouped[(r.video_id, r.agent_id, r.target_id)].append(
            (int(r.start_frame), int(r.stop_frame), r.action, int(r.row_id))
        )
    bad = 0
    for _key, intervals in grouped.items():
        intervals.sort()
        active: List[Tuple[int, int, str, int]] = []
        for s, e, a, rid in intervals:
            active = [iv for iv in active if iv[1] > s]
            for a_s, a_e, a_act, a_rid in active:
                if a_act != a:
                    bad += 1
                    print(f"  OVERLAP rows {a_rid} & {rid}: {a_act}[{a_s},{a_e}) "
                          f"vs {a}[{s},{e})")
            active.append((s, e, a, rid))
    if bad:
        raise SystemExit(
            f"[assert_no_pair_overlap] {bad} cross-action overlap(s) remain — "
            "submission would be rejected by Kaggle."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate MABe submission.csv (ensemble + TTA)")
    ap.add_argument("--model_paths", nargs="+", required=True,
                    help="One or more checkpoint paths; probs averaged before smoothing.")
    ap.add_argument("--test_csv", default=None)
    ap.add_argument("--test_tracking_dir", default=None)
    ap.add_argument("--train_csv", default=None,
                    help="Used with test.csv to build the global action list.")
    ap.add_argument("--action_list", default=None,
                    help="Path to action_list.npy; rebuilds from --train_csv if missing.")
    ap.add_argument("--output", default="submission.csv")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--min_run_len", type=int, default=1)
    ap.add_argument("--smooth_kind", choices=("none", "box", "gauss"), default="gauss",
                    help="Auto-overridden by --thresholds_json 'smoothing' field.")
    ap.add_argument("--smooth_k", type=float, default=5.0,
                    help="Box window or Gauss sigma; <=0 disables.")
    ap.add_argument("--tta_flip", action="store_true",
                    help="Extra forward pass with horizontal flip + L↔R body-part swap.")
    ap.add_argument("--tta_angles", type=float, nargs="*", default=[],
                    help="Rotation angles (deg) for TTA, e.g. --tta_angles -10 10.")
    ap.add_argument("--gap_tol", type=int, default=5,
                    help="Merge adjacent positive intervals with gap <= this; 0 disables.")
    ap.add_argument(
        "--single_action",
        dest="single_action",
        action="store_true",
        default=True,
        help="(default) At most one action per frame per (agent, target).",
    )
    ap.add_argument(
        "--multi_action",
        dest="single_action",
        action="store_false",
        help="Disable single-action enforcement.",
    )
    ap.add_argument("--stride", type=int, default=0,
                    help="0 uses window_size//2 per model (matches training overlap).")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--max_videos", type=int, default=0,
                    help="If >0, only process the first N test videos.")
    ap.add_argument("--thresholds_json", default=None,
                    help="best_thresholds.json; looks next to --model_paths[0] if omitted.")
    ap.add_argument("--no_tuned_thresholds", action="store_true")
    ap.add_argument("--features_mode", choices=("raw", "embed", "concat"),
                    default="raw",
                    help="raw | embed | concat. embed/concat require --embeddings_dir.")
    ap.add_argument("--embeddings_dir", type=str, default=None,
                    help="PoseBERT embedding cache for test videos.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    # Auto-resolve Kaggle paths when running on Kaggle.
    kaggle_root = find_kaggle_competition_root()
    if kaggle_root is not None:
        print(f"[kaggle] competition root: {kaggle_root}")
        if args.test_csv is None:
            cand = kaggle_root / "test.csv"
            if cand.is_file():
                args.test_csv = str(cand)
        if args.test_tracking_dir is None:
            cand = kaggle_root / "test_tracking"
            if cand.is_dir():
                args.test_tracking_dir = str(cand)
        if args.train_csv is None:
            cand = kaggle_root / "train.csv"
            if cand.is_file():
                args.train_csv = str(cand)

    missing = [name for name in ("test_csv", "test_tracking_dir")
               if getattr(args, name) is None]
    if missing:
        raise SystemExit(
            "Missing required inputs (pass via CLI or place under "
            f"/kaggle/input/<comp>/): {', '.join(missing)}"
        )

    device = get_device(args.device)
    print(f"Device: {device}")

    train_meta = pd.read_csv(args.train_csv) if args.train_csv else None
    test_meta = pd.read_csv(args.test_csv)
    test_meta["video_id"] = test_meta["video_id"].astype(str)

    if args.action_list and os.path.exists(args.action_list):
        actions = np.load(args.action_list, allow_pickle=True).tolist()
        print(f"Loaded {len(actions)} actions from {args.action_list}")
    elif train_meta is not None:
        union = pd.concat([train_meta, test_meta], ignore_index=True)
        actions, _, _ = build_action_universe(union)
        print(f"Built {len(actions)} actions from train+test")
    else:
        raise SystemExit("Need --action_list or --train_csv to determine actions.")

    action_to_idx = {a: i for i, a in enumerate(actions)}
    num_classes = len(actions)
    if num_classes != DEFAULT_MODEL_CONFIG["num_classes"]:
        print(
            f"WARNING: action list has {num_classes} but checkpoint expects "
            f"{DEFAULT_MODEL_CONFIG['num_classes']}. Output will use the first "
            f"{DEFAULT_MODEL_CONFIG['num_classes']} indices."
        )
    # Size the model to match the checkpoint; per-pair whitelist slices below.
    model_num_classes = DEFAULT_MODEL_CONFIG["num_classes"]

    _, per_video_whitelist, _ = build_action_universe(test_meta)

    embed_dim = 0
    if args.features_mode != "raw":
        if not args.embeddings_dir:
            raise SystemExit(f"--features_mode={args.features_mode} requires --embeddings_dir")
        import json as _json
        meta_path = os.path.join(args.embeddings_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise SystemExit(
                f"--embeddings_dir={args.embeddings_dir} is missing meta.json — "
                "run pose_bert.scripts.extract_embeddings on the test data first."
            )
        with open(meta_path) as fh:
            embed_dim = int(_json.load(fh)["d_model"])
        print(f"Features: mode={args.features_mode}  embeddings_dir={args.embeddings_dir}  "
              f"embed_dim={embed_dim}")
    if args.features_mode == "raw":
        model_input_dim = 176
    elif args.features_mode == "embed":
        model_input_dim = 2 * embed_dim
    else:
        model_input_dim = 176 + 2 * embed_dim

    # window_size auto-detected from each checkpoint.
    models: List[MABeTransformer] = []
    model_windows: List[int] = []
    for mp in args.model_paths:
        m = load_model(
            mp,
            num_classes=model_num_classes,
            device=device,
            window_size=None,
            input_dim=model_input_dim,
        )
        ws_m = int(m.encoder_layers[0].attention.rel_pos_bias.num_embeddings + 1) // 2
        models.append(m)
        model_windows.append(ws_m)
        print(f"Loaded model from {mp}  (window={ws_m})")
    print(f"[main] ensemble of {len(models)} model(s); windows={model_windows}")

    per_lab_action_thr: Dict[Tuple[str, int], float] = {}
    if not args.no_tuned_thresholds:
        thr_path = args.thresholds_json
        if thr_path is None:
            cand = os.path.join(
                os.path.dirname(os.path.abspath(args.model_paths[0])),
                "best_thresholds.json",
            )
            if os.path.exists(cand):
                thr_path = cand
        if thr_path and os.path.exists(thr_path):
            import json as _json
            with open(thr_path) as fh:
                thr_blob = _json.load(fh)
            entries = thr_blob.get("per_lab_action", [])
            for ent in entries:
                per_lab_action_thr[(str(ent["lab_id"]), int(ent["action_idx"]))] = float(ent["threshold"])
            print(f"[thresholds] loaded {len(per_lab_action_thr)} per-(lab,action) "
                  f"thresholds from {thr_path}")
            sm = thr_blob.get("smoothing")
            if sm and sm.get("kind") in ("box", "gauss") and sm.get("k") is not None:
                args.smooth_kind = "gauss" if sm["kind"] == "gauss" else "box"
                args.smooth_k = float(sm["k"])
                print(f"[smoothing] auto-loaded from JSON: kind={args.smooth_kind}, "
                      f"k={args.smooth_k}")
        else:
            print("[thresholds] no thresholds JSON found — using "
                  f"--threshold={args.threshold} for all (lab, action).")
    print(f"[smoothing] kind={args.smooth_kind}, k={args.smooth_k}")
    if args.tta_flip or args.tta_angles:
        print(f"[tta] flip={args.tta_flip}, angles={args.tta_angles}  "
              f"→ {1 + int(args.tta_flip) + len(args.tta_angles)} forward(s) per model per pair")

    # Vocabs come from train.csv ONLY (must match training); test-only values → UNK_IDX.
    pair_lookup: Dict | None = None
    needs_meta = any(bool(m.meta_embeddings) for m in models)
    if needs_meta:
        if not args.train_csv or not os.path.exists(args.train_csv):
            print("WARNING: model has metadata layers but --train_csv is "
                  "missing — every sample will map to UNK. Predictions will "
                  "be degraded. Provide train.csv to fix.")
        else:
            print(f"Building metadata vocabs from {args.train_csv}")
            vocabs = build_vocabs(args.train_csv)
            print(f"  vocab sizes (lab, strain, sex) = "
                  f"{vocab_sizes_per_table(vocabs)}")
            # Build lookup over train ∪ test rows so test (video, agent, target) keys resolve.
            import tempfile
            with tempfile.NamedTemporaryFile(
                "w", suffix=".csv", delete=False
            ) as fh:
                combined_path = fh.name
            try:
                tr = pd.read_csv(args.train_csv)
                te = pd.read_csv(args.test_csv)
                pd.concat([tr, te], ignore_index=True).to_csv(
                    combined_path, index=False
                )
                pair_lookup = build_pair_lookup(combined_path, vocabs)
                print(f"  built pair lookup over train+test: "
                      f"{len(pair_lookup)} (video, agent, target) entries")
            finally:
                try:
                    os.unlink(combined_path)
                except OSError:
                    pass

    test_rows = list(test_meta.itertuples(index=False))
    if args.max_videos > 0:
        test_rows = test_rows[: args.max_videos]

    submission_rows: List[Dict] = []

    for row in tqdm(test_rows, desc="Test videos"):
        lab_id = getattr(row, "lab_id")
        video_id = str(getattr(row, "video_id"))

        pair_whitelist = per_video_whitelist.get((lab_id, video_id), {})
        if not pair_whitelist:
            continue

        tracking_path = os.path.join(args.test_tracking_dir, lab_id,
                                     f"{video_id}.parquet")
        if not os.path.exists(tracking_path):
            print(f"  MISSING tracking for {lab_id}/{video_id}, skipping.")
            continue

        fps = float(getattr(row, "frames_per_second", 25.0) or 25.0)
        pix_per_cm = float(getattr(row, "pix_per_cm_approx", 1.0) or 1.0)

        tracking_df = pd.read_parquet(tracking_path)
        tracking_df = tracking_df[~tracking_df["bodypart"].isin(DROP_PARTS)]
        if tracking_df.empty:
            continue

        # Pre-load every per-mouse embedding cache for this video once; pairs slice.
        mouse_emb: Dict[int, np.ndarray] = {}
        if args.features_mode != "raw":
            emb_video_dir = os.path.join(args.embeddings_dir, lab_id, video_id)
            unique_mouse_ids = set()
            for (a, t) in pair_whitelist.keys():
                unique_mouse_ids.add(int(a)); unique_mouse_ids.add(int(t))
            missing = []
            for mid in unique_mouse_ids:
                p = os.path.join(emb_video_dir, f"{mid}.npy")
                if os.path.exists(p):
                    mouse_emb[mid] = np.load(p, mmap_mode='r')
                else:
                    missing.append(mid)
            if missing:
                print(f"  MISSING embeddings for {lab_id}/{video_id} mice "
                      f"{missing} — skipping this video.")
                continue

        video_frames = sorted(tracking_df["video_frame"].unique())
        vf = np.array(video_frames, dtype=np.int64)
        video_duration_sec = float(getattr(row, "video_duration_sec", 0.0) or 0.0)
        if video_duration_sec > 0:
            total_frames = int(round(video_duration_sec * fps))
        else:
            total_frames = int(vf[-1]) + 1
        total_frames = max(1, total_frames)

        for (agent_id, target_id), allowed_actions in pair_whitelist.items():
            try:
                agent_feat_df, target_feat_df = process_directed_pair_agent_centric(
                    tracking_df=tracking_df,
                    agent_id=agent_id,
                    target_id=target_id,
                    fps=fps,
                    pix_per_cm=pix_per_cm,
                    parts_order=ALL_PARTS,
                    drop_parts=DROP_PARTS,
                    video_frames=video_frames,
                    bodyparts_already_dropped=True,
                )
            except Exception as e:
                print(f"  SKIP {lab_id}/{video_id} pair "
                      f"({agent_id},{target_id}): {e}")
                continue

            raw_features = np.concatenate(
                [agent_feat_df.values, target_feat_df.values], axis=1
            ).astype(np.float32)
            if args.features_mode == "raw":
                features = raw_features
            else:
                a_emb = np.asarray(mouse_emb[int(agent_id)], dtype=np.float32)
                t_emb = np.asarray(mouse_emb[int(target_id)], dtype=np.float32)
                T_align = min(raw_features.shape[0], a_emb.shape[0], t_emb.shape[0])
                if args.features_mode == "embed":
                    features = np.concatenate(
                        [a_emb[:T_align], t_emb[:T_align]], axis=1
                    ).astype(np.float32)
                else:
                    features = np.concatenate(
                        [raw_features[:T_align], a_emb[:T_align], t_emb[:T_align]],
                        axis=1,
                    ).astype(np.float32)

            meta_ids = None
            if pair_lookup is not None:
                meta_ids = resolve_meta_ids(
                    pair_lookup, video_id, int(agent_id), int(target_id)
                )

            tta_specs: List[Tuple[bool, float]] = [(False, 0.0)]
            if args.tta_flip:
                tta_specs.append((True, 0.0))
            for ang in args.tta_angles:
                tta_specs.append((False, float(ang)))

            n_passes = len(models) * len(tta_specs)
            probs_sum = None
            for m_idx, m in enumerate(models):
                ws_m = model_windows[m_idx]
                stride_m = args.stride if args.stride > 0 else max(1, ws_m // 2)
                for flip, ang in tta_specs:
                    feats_v = (features if not flip and ang == 0.0
                               else tta_transform(features, flip=flip, angle_deg=ang))
                    p = predict_video_pair(
                        model=m,
                        features=feats_v,
                        device=device,
                        window_size=ws_m,
                        stride=stride_m,
                        batch_size=args.batch_size,
                        meta_ids=meta_ids,
                    )
                    probs_sum = p if probs_sum is None else (probs_sum + p)
            probs = probs_sum / float(n_passes)
            probs = smooth_probs(probs, args.smooth_kind, args.smooth_k)

            agent_str = f"mouse{int(agent_id)}"
            # Self-directed (agent == target) requires the literal "self" target_id.
            if int(agent_id) == int(target_id):
                target_str = "self"
            else:
                target_str = f"mouse{int(target_id)}"

            allowed_action_idx: Dict[str, int] = {}
            for action in allowed_actions:
                a_idx = action_to_idx.get(action)
                if a_idx is not None and a_idx < probs.shape[1]:
                    allowed_action_idx[action] = a_idx

            if not allowed_action_idx:
                continue

            # Per-frame argmax over allowed actions; enforce_pair_no_overlap catches close_mask leakage.
            winner_idx: np.ndarray | None = None
            if args.single_action:
                idx_array = np.array(sorted(allowed_action_idx.values()),
                                     dtype=np.int64)
                restricted = probs[:, idx_array]
                winner_local = np.argmax(restricted, axis=1)
                winner_idx = idx_array[winner_local]

            for action, a_idx in sorted(allowed_action_idx.items()):
                action_probs = probs[:, a_idx]
                thr = per_lab_action_thr.get((lab_id, a_idx), args.threshold)
                mask = action_probs > thr
                if winner_idx is not None:
                    mask &= (winner_idx == a_idx)
                if not mask.any():
                    continue

                # close_mask CAN produce cross-action overlaps; enforce_pair_no_overlap fixes those.
                mask = close_mask(mask, args.gap_tol)

                for start_frame, stop_frame, run_s, run_e in runs_from_mask(
                    mask, vf, args.min_run_len
                ):
                    s = max(0, int(start_frame))
                    e = min(int(total_frames), int(stop_frame))
                    if e <= s:
                        continue
                    # Mean probability over the run = confidence (used by overlap resolver).
                    confidence = float(action_probs[run_s:run_e].mean())
                    submission_rows.append({
                        "video_id": video_id,
                        "agent_id": agent_str,
                        "target_id": target_str,
                        "action": action,
                        "start_frame": s,
                        "stop_frame": e,
                        "confidence": confidence,
                    })

    submission_rows = enforce_pair_no_overlap(submission_rows)

    expected_cols = ["row_id", "video_id", "agent_id", "target_id",
                     "action", "start_frame", "stop_frame"]

    if submission_rows:
        sub_df = pd.DataFrame(submission_rows)
        sub_df.insert(0, "row_id", np.arange(len(sub_df), dtype=np.int64))
    else:
        # Trivial fallback row to satisfy Kaggle's non-empty-file requirement.
        # Pull from THIS run's test whitelist — do NOT copy sample_submission.csv
        # since its video_id is the public test video, not the hidden test set.
        print("WARNING: model produced zero rows — emitting 1 trivial fallback "
              "row to satisfy Kaggle's non-empty-file requirement.")
        fb_row: Dict | None = None
        for (lab_id, vid), pair_dict in per_video_whitelist.items():
            for (agent_id, target_id), allowed in pair_dict.items():
                if allowed:
                    fb_target = (
                        "self" if int(agent_id) == int(target_id)
                        else f"mouse{int(target_id)}"
                    )
                    fb_row = {
                        "video_id": vid,
                        "agent_id": f"mouse{int(agent_id)}",
                        "target_id": fb_target,
                        "action": sorted(allowed)[0],
                        "start_frame": 0,
                        "stop_frame": 1,
                    }
                    break
            if fb_row is not None:
                break
        if fb_row is not None:
            sub_df = pd.DataFrame([fb_row])
            sub_df.insert(0, "row_id", np.arange(len(sub_df), dtype=np.int64))
        else:
            sub_df = pd.DataFrame(columns=expected_cols)

    # Drop internal-only `confidence` column before writing.
    sub_df = sub_df[[c for c in expected_cols if c in sub_df.columns]].copy()

    sub_df["video_id"] = pd.to_numeric(sub_df["video_id"], errors="coerce").astype("Int64")
    sub_df["start_frame"] = pd.to_numeric(sub_df["start_frame"], errors="coerce").astype("Int64")
    sub_df["stop_frame"] = pd.to_numeric(sub_df["stop_frame"], errors="coerce").astype("Int64")
    for col in ("agent_id", "target_id", "action"):
        sub_df[col] = sub_df[col].astype(str)

    sub_df = sub_df.dropna(subset=["video_id", "agent_id", "target_id",
                                   "action", "start_frame", "stop_frame"])
    sub_df = sub_df[
        sub_df["stop_frame"].astype(np.int64) > sub_df["start_frame"].astype(np.int64)
    ]
    sub_df = sub_df.reset_index(drop=True)

    sub_df["row_id"] = np.arange(len(sub_df), dtype=np.int64)
    sub_df["video_id"] = sub_df["video_id"].astype(np.int64)
    sub_df["start_frame"] = sub_df["start_frame"].astype(np.int64)
    sub_df["stop_frame"] = sub_df["stop_frame"].astype(np.int64)

    if len(sub_df) == 0:
        raise SystemExit(
            "Refusing to write an empty submission (would be rejected by "
            "Kaggle). The fallback also emitted zero rows — check that "
            "test.csv is non-empty and behaviors_labeled is populated."
        )

    assert_no_pair_overlap(sub_df)

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sub_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(sub_df)} rows to {out_path}")
    print(sub_df.head())
    print(sub_df.dtypes)


if __name__ == "__main__":
    main()
