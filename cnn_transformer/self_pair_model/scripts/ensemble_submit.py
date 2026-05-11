"""Dual-head ensembled inference and Kaggle submission writer.

Usage:
    python scripts/ensemble_submit.py --pair_model_paths PAIR1.pt PAIR2.pt \\
        --self_model_paths SELF1.pt SELF2.pt --output submission.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
_MODEL_DIR = _ROOT / "model"
_PREP_DIR = _ROOT.parent.parent / "preprocessing"
sys.path.insert(0, str(_MODEL_DIR))
sys.path.insert(0, str(_PREP_DIR))

from model import MABeTransformer  # noqa: E402

from inference import (  # noqa: E402
    DEFAULT_MODEL_CONFIG,
    WINDOW_SIZE,
    STRIDE,
    find_kaggle_competition_root,
    get_device,
    load_model,
    predict_video_pair,
    smooth_probs,
    close_mask,
    runs_from_mask,
    enforce_pair_no_overlap,
    assert_no_pair_overlap,
    _load_per_lab_thresholds_json,
)

from preprocess_features import (  # noqa: E402
    ALL_PARTS,
    DROP_PARTS,
    process_directed_pair_agent_centric,
)
from preprocess_data import build_action_universe  # noqa: E402
from action_taxonomy import PAIR_ACTIONS, SELF_ACTIONS  # noqa: E402
from metadata import (  # noqa: E402
    build_pair_lookup,
    build_vocabs,
    resolve_meta_ids,
    vocab_sizes_per_table,
)


def _model_window_size(m: MABeTransformer) -> int:
    if not hasattr(m, "encoder_layers") or len(m.encoder_layers) == 0:
        raise SystemExit("[ensemble_submit] loaded model has no encoder_layers.")
    rows = int(m.encoder_layers[0].attention.rel_pos_bias.num_embeddings)
    return (rows + 1) // 2


def _model_input_dim(m: MABeTransformer) -> int:
    return int(m.input_projection[0].in_features)


def _validate_head(
    models: List[MABeTransformer],
    head_name: str,
    expected_input_dim: int,
    expected_num_classes: int,
    paths: List[str],
) -> None:
    for m, p in zip(models, paths):
        idim = _model_input_dim(m)
        ncls = int(m.num_classes)
        if idim != expected_input_dim or ncls != expected_num_classes:
            raise SystemExit(
                f"[{head_name}] checkpoint {p} has input_dim={idim} num_classes={ncls}; "
                f"expected input_dim={expected_input_dim} num_classes={expected_num_classes}."
            )


def _normalize_weights(weights: List[float] | None, n: int, head_name: str) -> np.ndarray:
    if weights is None or len(weights) == 0:
        return np.full(n, 1.0 / max(n, 1), dtype=np.float64)
    if len(weights) != n:
        raise SystemExit(
            f"--{head_name}_weights has {len(weights)} entries but {n} {head_name} models loaded."
        )
    w = np.asarray(weights, dtype=np.float64)
    s = w.sum()
    if s <= 0:
        raise SystemExit(f"--{head_name}_weights must sum to > 0.")
    return w / s


@torch.no_grad()
def ensemble_predict(
    models: List[MABeTransformer],
    weights: np.ndarray,
    features: np.ndarray,
    device: torch.device,
    stride: int,
    batch_size: int,
    meta_ids: np.ndarray | None,
) -> np.ndarray:
    accum: np.ndarray | None = None
    for m, w in zip(models, weights):
        ws = _model_window_size(m)
        probs = predict_video_pair(
            model=m,
            features=features,
            device=device,
            window_size=ws,
            stride=stride,
            batch_size=batch_size,
            meta_ids=meta_ids,
        )
        if accum is None:
            accum = (float(w) * probs).astype(np.float32)
        else:
            accum += (float(w) * probs).astype(np.float32)
    assert accum is not None
    return accum


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MABe dual-head ensembled submission.")
    ap.add_argument("--pair_model_paths", nargs="*", default=[],
                    help="Pair-head checkpoints (zero or more).")
    ap.add_argument("--self_model_paths", nargs="*", default=[],
                    help="Self-head checkpoints (zero or more).")
    ap.add_argument("--pair_weights", type=float, nargs="*", default=None,
                    help="Optional per-pair-model weights. Default: uniform.")
    ap.add_argument("--self_weights", type=float, nargs="*", default=None,
                    help="Optional per-self-model weights. Default: uniform.")
    ap.add_argument("--pair_thresholds_json", default=None,
                    help="thresholds JSON from ensemble_eval.py --task pair.")
    ap.add_argument("--self_thresholds_json", default=None,
                    help="thresholds JSON from ensemble_eval.py --task self.")

    ap.add_argument("--test_csv", default=None)
    ap.add_argument("--test_tracking_dir", default=None)
    ap.add_argument("--train_csv", default=None,
                    help="Used to build the metadata vocab and pair lookup.")

    ap.add_argument("--output", default="submission.csv")
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="Fallback threshold when a (lab, action) is missing from JSON.")
    ap.add_argument("--min_run_len", type=int, default=1)
    ap.add_argument("--smooth_window", type=int, default=5,
                    help="Box smoothing of averaged per-head probabilities.")
    ap.add_argument("--pair_smooth_window", type=int, default=None,
                    help="Per-head override for the pair branch.")
    ap.add_argument("--self_smooth_window", type=int, default=None,
                    help="Per-head override for the self branch.")
    ap.add_argument("--gap_tol", type=int, default=5)
    ap.add_argument(
        "--single_action",
        dest="single_action",
        action="store_true",
        default=True,
        help="(default) at most one action per frame per (agent, target) via argmax.",
    )
    ap.add_argument(
        "--multi_action",
        dest="single_action",
        action="store_false",
        help="Disable single-action enforcement.",
    )

    ap.add_argument("--stride", type=int, default=STRIDE,
                    help="Sliding-window stride at inference.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--max_videos", type=int, default=0,
                    help="If >0, only process the first N test videos.")
    ap.add_argument("--no_tuned_thresholds", action="store_true",
                    help="Ignore thresholds JSONs and use --threshold uniformly.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

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
        if args.train_csv and not os.path.isfile(args.train_csv):
            cand = kaggle_root / "train.csv"
            if cand.is_file():
                args.train_csv = str(cand)
        if args.test_csv and not os.path.isfile(args.test_csv):
            cand = kaggle_root / "test.csv"
            if cand.is_file():
                args.test_csv = str(cand)
        if args.test_tracking_dir and not os.path.isdir(args.test_tracking_dir):
            cand = kaggle_root / "test_tracking"
            if cand.is_dir():
                args.test_tracking_dir = str(cand)

    missing = [name for name in ("test_csv", "test_tracking_dir")
               if getattr(args, name) is None]
    if missing:
        raise SystemExit(
            "Missing required inputs (pass via CLI or place under "
            f"/kaggle/input/<comp>/): {', '.join(missing)}"
        )

    pair_smooth = (args.pair_smooth_window
                   if args.pair_smooth_window is not None else args.smooth_window)
    self_smooth = (args.self_smooth_window
                   if args.self_smooth_window is not None else args.smooth_window)
    print(f"[smoothing] pair K={pair_smooth}  self K={self_smooth}")

    if not args.pair_model_paths and not args.self_model_paths:
        raise SystemExit("Provide at least one of --pair_model_paths / --self_model_paths.")
    if args.pair_model_paths and not args.pair_thresholds_json and not args.no_tuned_thresholds:
        print("WARNING: pair models given without --pair_thresholds_json — "
              f"using fallback --threshold={args.threshold} for all (lab, action).")
    if args.self_model_paths and not args.self_thresholds_json and not args.no_tuned_thresholds:
        print("WARNING: self models given without --self_thresholds_json — "
              f"using fallback --threshold={args.threshold} for all (lab, action).")

    device = get_device(args.device)
    print(f"Device: {device}")

    print(f"[ensemble] pair_models={len(args.pair_model_paths)} "
          f"self_models={len(args.self_model_paths)}")

    pair_actions = list(PAIR_ACTIONS)
    self_actions = list(SELF_ACTIONS)
    pair_action_to_idx = {a: i for i, a in enumerate(pair_actions)}
    self_action_to_idx = {a: i for i, a in enumerate(self_actions)}

    test_meta = pd.read_csv(args.test_csv)
    test_meta["video_id"] = test_meta["video_id"].astype(str)
    _, per_video_whitelist, _ = build_action_universe(test_meta)

    pair_models: List[MABeTransformer] = []
    for p in args.pair_model_paths:
        m = load_model(
            p,
            num_classes=len(PAIR_ACTIONS),
            device=device,
            input_dim=176,
        )
        pair_models.append(m)
        print(f"[load_pair] {p} -> input_dim={_model_input_dim(m)} "
              f"num_classes={int(m.num_classes)} window={_model_window_size(m)}")

    self_models: List[MABeTransformer] = []
    for p in args.self_model_paths:
        m = load_model(
            p,
            num_classes=len(SELF_ACTIONS),
            device=device,
            input_dim=88,
        )
        self_models.append(m)
        print(f"[load_self] {p} -> input_dim={_model_input_dim(m)} "
              f"num_classes={int(m.num_classes)} window={_model_window_size(m)}")

    _validate_head(pair_models, "pair", 176, len(PAIR_ACTIONS), args.pair_model_paths)
    _validate_head(self_models, "self", 88, len(SELF_ACTIONS), args.self_model_paths)

    pair_w = _normalize_weights(args.pair_weights, len(pair_models), "pair")
    self_w = _normalize_weights(args.self_weights, len(self_models), "self")
    if len(pair_models):
        print(f"[ensemble] pair weights: {np.round(pair_w, 4).tolist()}")
    if len(self_models):
        print(f"[ensemble] self weights: {np.round(self_w, 4).tolist()}")

    per_lab_pair_thr: Dict[Tuple[str, int], float] = {}
    per_lab_self_thr: Dict[Tuple[str, int], float] = {}
    if not args.no_tuned_thresholds:
        if args.pair_thresholds_json and os.path.isfile(args.pair_thresholds_json):
            per_lab_pair_thr = _load_per_lab_thresholds_json(args.pair_thresholds_json)
            print(f"[thresholds] pair head: loaded {len(per_lab_pair_thr)} per-(lab,action) "
                  f"from {args.pair_thresholds_json}")
        elif args.pair_model_paths:
            print(f"[thresholds] pair head: no JSON — using --threshold={args.threshold}")
        if args.self_thresholds_json and os.path.isfile(args.self_thresholds_json):
            per_lab_self_thr = _load_per_lab_thresholds_json(args.self_thresholds_json)
            print(f"[thresholds] self head: loaded {len(per_lab_self_thr)} per-(lab,action) "
                  f"from {args.self_thresholds_json}")
        elif args.self_model_paths:
            print(f"[thresholds] self head: no JSON — using --threshold={args.threshold}")

    needs_meta = any(len(m.meta_embeddings) > 0 for m in pair_models + self_models)
    pair_lookup: Dict | None = None
    if needs_meta:
        if not args.train_csv or not os.path.isfile(args.train_csv):
            print("WARNING: at least one model uses metadata embeddings but --train_csv is "
                  "missing — every sample will map to UNK and predictions will be degraded.")
        else:
            print(f"Building metadata vocabs from {args.train_csv}")
            vocabs = build_vocabs(args.train_csv)
            print(f"  vocab sizes (lab, strain, sex) = {vocab_sizes_per_table(vocabs)}")
            import tempfile
            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
                combined_path = fh.name
            try:
                tr = pd.read_csv(args.train_csv)
                te = pd.read_csv(args.test_csv)
                pd.concat([tr, te], ignore_index=True).to_csv(combined_path, index=False)
                pair_lookup = build_pair_lookup(combined_path, vocabs)
                print(f"  built pair lookup over train+test: {len(pair_lookup)} entries")
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

        tracking_path = os.path.join(args.test_tracking_dir, lab_id, f"{video_id}.parquet")
        if not os.path.exists(tracking_path):
            print(f"  MISSING tracking for {lab_id}/{video_id}, skipping.")
            continue

        fps = float(getattr(row, "frames_per_second", 25.0) or 25.0)
        pix_per_cm = float(getattr(row, "pix_per_cm_approx", 1.0) or 1.0)

        tracking_df = pd.read_parquet(tracking_path)
        tracking_df = tracking_df[~tracking_df["bodypart"].isin(DROP_PARTS)]
        if tracking_df.empty:
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
                print(f"  SKIP {lab_id}/{video_id} pair ({agent_id},{target_id}): {e}")
                continue

            pair_features = np.concatenate(
                [agent_feat_df.values, target_feat_df.values], axis=1
            ).astype(np.float32)
            self_features = agent_feat_df.values.astype(np.float32)

            meta_ids = None
            if pair_lookup is not None:
                meta_ids = resolve_meta_ids(
                    pair_lookup, video_id, int(agent_id), int(target_id)
                )

            if pair_models and int(agent_id) != int(target_id):
                p_actions = [a for a in allowed_actions if a in pair_action_to_idx]
                if p_actions:
                    p_probs = ensemble_predict(
                        models=pair_models,
                        weights=pair_w,
                        features=pair_features,
                        device=device,
                        stride=args.stride,
                        batch_size=args.batch_size,
                        meta_ids=meta_ids,
                    )
                    p_probs = smooth_probs(p_probs, pair_smooth)

                    agent_str = f"mouse{int(agent_id)}"
                    target_str = f"mouse{int(target_id)}"

                    allowed_action_idx = {a: pair_action_to_idx[a] for a in p_actions}
                    winner_idx: np.ndarray | None = None
                    if args.single_action:
                        idx_array = np.array(sorted(allowed_action_idx.values()), dtype=np.int64)
                        restricted = p_probs[:, idx_array]
                        winner_local = np.argmax(restricted, axis=1)
                        winner_idx = idx_array[winner_local]

                    for action, a_idx in sorted(allowed_action_idx.items()):
                        action_probs = p_probs[:, a_idx]
                        thr = per_lab_pair_thr.get((str(lab_id), a_idx), args.threshold)
                        mask = action_probs > thr
                        if winner_idx is not None:
                            mask &= (winner_idx == a_idx)
                        if not mask.any():
                            continue
                        mask = close_mask(mask, args.gap_tol)
                        for start_frame, stop_frame, run_s, run_e in runs_from_mask(
                            mask, vf, args.min_run_len
                        ):
                            s = max(0, int(start_frame))
                            e = min(int(total_frames), int(stop_frame))
                            if e <= s:
                                continue
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

            if self_models and int(agent_id) == int(target_id):
                s_actions = [a for a in allowed_actions if a in self_action_to_idx]
                if not s_actions:
                    continue
                s_probs = ensemble_predict(
                    models=self_models,
                    weights=self_w,
                    features=self_features,
                    device=device,
                    stride=args.stride,
                    batch_size=args.batch_size,
                    meta_ids=meta_ids,
                )
                s_probs = smooth_probs(s_probs, self_smooth)

                agent_str = f"mouse{int(agent_id)}"
                target_str = "self"
                allowed_action_idx = {a: self_action_to_idx[a] for a in s_actions}
                winner_idx: np.ndarray | None = None
                if args.single_action:
                    idx_array = np.array(sorted(allowed_action_idx.values()), dtype=np.int64)
                    restricted = s_probs[:, idx_array]
                    winner_local = np.argmax(restricted, axis=1)
                    winner_idx = idx_array[winner_local]

                for action, a_idx in sorted(allowed_action_idx.items()):
                    action_probs = s_probs[:, a_idx]
                    thr = per_lab_self_thr.get((str(lab_id), a_idx), args.threshold)
                    mask = action_probs > thr
                    if winner_idx is not None:
                        mask &= (winner_idx == a_idx)
                    if not mask.any():
                        continue
                    mask = close_mask(mask, args.gap_tol)
                    for start_frame, stop_frame, run_s, run_e in runs_from_mask(
                        mask, vf, args.min_run_len
                    ):
                        s = max(0, int(start_frame))
                        e = min(int(total_frames), int(stop_frame))
                        if e <= s:
                            continue
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
        # Fallback row uses a whitelist video_id (not sample_submission's, which Kaggle rejects).
        print("WARNING: ensemble produced zero rows — emitting 1 trivial fallback row.")
        fb_row: Dict | None = None
        for (lab_id_fb, vid), pair_dict in per_video_whitelist.items():
            for (a_id, t_id), allowed in pair_dict.items():
                if allowed:
                    fb_target = (
                        "self" if int(a_id) == int(t_id) else f"mouse{int(t_id)}"
                    )
                    fb_row = {
                        "video_id": vid,
                        "agent_id": f"mouse{int(a_id)}",
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
            "Refusing to write an empty submission (would be rejected by Kaggle). "
            "The fallback also emitted zero rows — check that test.csv is non-empty "
            "and behaviors_labeled is populated."
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
