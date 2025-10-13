import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score

from detectors.ppl import PPLDetector
from detectors.self_critique import SelfCritiqueDetector

def evaluate_performance_common(y_true, y_scores):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    is_finite = np.isfinite(y_scores)
    y_true = y_true[is_finite]
    y_scores = y_scores[is_finite]

    if len(np.unique(y_true)) < 2:
        return {
            "roc_auc": np.nan,
            "best_f1_score": np.nan,
            "accuracy_at_best_f1": np.nan,
            "optimal_threshold_f1": np.nan,
            "youden_j_score": np.nan,
            "optimal_threshold_youden": np.nan,
            "f1_at_youden_threshold": np.nan,
            "accuracy_at_youden_threshold": np.nan,
            "tpr_at_fpr_5": np.nan,
            "error": "Only one class present."
        }

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    fscore = (2 * precision * recall) / (precision + recall + 1e-6)
    best_f1_idx = np.argmax(fscore[:-1]) if len(fscore) > 1 else 0
    optimal_threshold_f1 = pr_thresholds[best_f1_idx]
    best_f1 = fscore[best_f1_idx]
    y_pred_f1 = (y_scores >= optimal_threshold_f1).astype(int)
    accuracy_at_best_f1 = accuracy_score(y_true, y_pred_f1)

    youden_j_scores = tpr - fpr
    best_youden_idx = np.argmax(youden_j_scores)
    youden_j_score = youden_j_scores[best_youden_idx]
    optimal_threshold_youden = roc_thresholds[best_youden_idx]

    y_pred_youden = (y_scores >= optimal_threshold_youden).astype(int)
    tp_youden = np.sum((y_true == 1) & (y_pred_youden == 1))
    fp_youden = np.sum((y_true == 0) & (y_pred_youden == 1))
    fn_youden = np.sum((y_true == 1) & (y_pred_youden == 0))
    precision_youden = tp_youden / (tp_youden + fp_youden + 1e-6)
    recall_youden = tp_youden / (tp_youden + fn_youden + 1e-6)
    f1_at_youden_threshold = (2 * precision_youden * recall_youden) / (precision_youden + recall_youden + 1e-6)
    accuracy_at_youden_threshold = accuracy_score(y_true, y_pred_youden)

    target_fpr = 0.05
    indices_above_target = np.where(fpr >= target_fpr)[0]
    if len(indices_above_target) > 0:
        target_idx = indices_above_target[0] - 1 if indices_above_target[0] > 0 else 0
        tpr_at_fpr_5 = tpr[target_idx]
    else:
        tpr_at_fpr_5 = tpr[-1] if len(tpr) > 0 else np.nan

    return {
        "roc_auc": roc_auc,
        "best_f1_score": best_f1,
        "accuracy_at_best_f1": accuracy_at_best_f1,
        "optimal_threshold_f1": optimal_threshold_f1,
        "youden_j_score": youden_j_score,
        "optimal_threshold_youden": optimal_threshold_youden,
        "f1_at_youden_threshold": f1_at_youden_threshold,
        "accuracy_at_youden_threshold": accuracy_at_youden_threshold,
        "tpr_at_fpr_5": tpr_at_fpr_5
    }


def evaluate_on_dataframe(df_scores, detectors):
    """
    Evaluate all methods on given DataFrame (can be full dataset or subset).
    Returns a dictionary: overall + mean AUC + breakdown by source for each method.
    """
    final_evaluation = {}
    for detector in detectors:
        method_name = detector.get_name()
        direction = detector.get_direction()

        if method_name not in df_scores.columns:
            continue

        df_method = df_scores.dropna(subset=[method_name])
        y_true = df_method['ground_truth_label'].values
        y_scores = df_method[method_name].values * direction

        overall_perf = evaluate_performance_common(y_true, y_scores)

        breakdown = {}
        valid_aucs = []
        for source, group in df_method.groupby('data_source'):
            group_perf = evaluate_performance_common(
                group['ground_truth_label'].values,
                group[method_name].values * direction
            )
            breakdown[source] = group_perf
            if not np.isnan(group_perf['roc_auc']):
                valid_aucs.append(group_perf['roc_auc'])

        mean_auc = float(np.mean(valid_aucs)) if valid_aucs else np.nan
        final_evaluation[method_name] = {
            "overall_performance": overall_perf,
            "mean_auc": mean_auc,
            "breakdown_by_source": breakdown
        }
    return final_evaluation

def build_all_detectors():
    detectors = [
        PPLDetector(),
        SelfCritiqueDetector(),
    ]
    return detectors

def detector_registry(detectors):
    """
    Build a dictionary of {alias: detector_instance} for precise targeting via --subset_by_detector.
    Alias definitions:
      ppl    -> PPLDetector
    """
    reg = {}
    for d in detectors:
        name = d.get_name().lower()
        # Simple alias mapping based on type/attributes
        if isinstance(d, PPLDetector):
            reg['ppl'] = d
    return reg


def main():
    parser = argparse.ArgumentParser(description="Calculate performance for all modular detection methods.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_summary_json", type=str, required=True)
    parser.add_argument("--output_plot", type=str, required=True)
    parser.add_argument("--output_dime_detail_jsonl", type=str, required=False)
    parser.add_argument("--output_self_critique_y_scores", type=str, default=None)

    parser.add_argument("--subset_by_detector", type=str, default=None,choices=["ppl"])
    parser.add_argument("--subset_quantile", type=float, default=0.5,
                    help="Quantile point for subset division (default 0.5, i.e., half).")
    parser.add_argument("--subset_take", type=str, default="low",
                    choices=["low", "high"],
                    help="Take samples below (low) or above (high) the quantile point, default low.")

    parser.add_argument("--output_summary_json_subset", type=str, default=None,
                        help="JSON output path for subset evaluation.")
    parser.add_argument("--output_plot_subset", type=str, default=None,
                        help="ROC plot output path for subset evaluation.")

    # Random control subset (equal proportion random sampling)
    parser.add_argument("--with_random_control", action="store_true",
                        help="If given, perform equal proportion random control evaluation on the selected subset sample ratio.")
    parser.add_argument("--random_seed", type=int, default=2025,
                        help="Random seed for random control subset.")
    parser.add_argument("--output_summary_json_random", type=str, default=None,
                        help="Random control subset evaluation JSON output path (if not provided, will be auto-generated next to subset path).")
    parser.add_argument("--output_plot_random", type=str, default=None,
                        help="Random control subset ROC plot output path (if not provided, will be auto-generated next to subset path).")

    args = parser.parse_args()

    # 1) Initialize detectors
    detectors = build_all_detectors()
    reg = detector_registry(detectors)  # For accessing pre-trained detectors by alias

    # 2) Read data and calculate scores for each method
    all_scores_list = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing samples"):
            data_item = json.loads(line)
            scores = {
                "ground_truth_label": data_item['ground_truth_label'],
                "data_source": data_item.get('data_source', 'unknown'),
                "original_user_content": data_item.get('original_user_content')
            }
            for detector in detectors:
                try:
                    scores[detector.get_name()] = detector.calculate_score(data_item)
                except Exception as e:
                    # Record NaN on exception to avoid losing entire sample
                    scores[detector.get_name()] = np.nan
            all_scores_list.append(scores)
    df_scores = pd.DataFrame(all_scores_list)

    # === After df_scores construction, before evaluation ===
    df_full = df_scores  # Keep full dataset

    def _pick_subset_by_detector(df, detectors, det_key, q, take="low"):
        """
        det_key: 'ppl'
        Returns: (df_subset, det_col, det_dir, keep_ratio)
        """
        det_key = (det_key or "").lower().strip()
        target = None
        for d in detectors:
            print('d', d)
            name = d.get_name().lower()
            print('name ', name)
            if det_key == "ppl" and "perplexity" in name:
                target = d; break
        if target is None:
            raise ValueError(f"[subset] No matching detector found: {det_key}.")

        det_col = target.get_name()
        det_dir = target.get_direction()

        if det_col not in df.columns:
            raise ValueError(f"[subset] Column {det_col} not in score table, please ensure the detector has been calculated.")

        # Align direction and calculate quantile only on finite values
        arr_all = (df[det_col].astype(float).to_numpy() * det_dir)
        finite_mask_all = np.isfinite(arr_all)
        arr = arr_all[finite_mask_all]
        if arr.size == 0:
            raise ValueError(f"[subset] Detector {det_col} scores are all NaN/Inf, cannot calculate quantile.")

        q = float(q)
        if not (0.0 < q <= 1.0):
            raise ValueError(f"[subset] subset_quantile must be in (0,1], current={q}")

        thr = np.quantile(arr, q)

        # Generate mask: compare only on finite values; set all non-finite values to False (not selected)
        if take == "low":
            pick_mask_finite = (arr_all <= thr)
        else:
            pick_mask_finite = (arr_all >= thr)
        mask = np.logical_and(pick_mask_finite, np.isfinite(arr_all))

        df_subset = df.loc[mask].copy()
        keep_ratio = float(df_subset.shape[0]) / max(1, df.shape[0])

        print(f"[subset] detector={det_key} -> col='{det_col}', dir={det_dir:+d}, "
            f"take={take}, q={q:.2f}, thr={thr:.6f}, keep_ratio={keep_ratio:.3f}, "
            f"n={len(df_subset)}/{len(df)}")

        return df_subset, det_col, det_dir, keep_ratio



    def _random_subset(df, keep_ratio, seed=2025):
        """Random sampling by proportion, returns random control subset of same scale."""
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"[random] keep_ratio must be in (0,1], current={keep_ratio}")
        n = df.shape[0]
        k = max(1, int(round(n * keep_ratio)))
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=k, replace=False)
        return df.iloc[idx].copy()


    # ========== Optional: quantile-based "subset evaluation" + "random control evaluation" ==========
    df_eval_main = df_full   # Default: full dataset evaluation
    subset_used = False
    random_used = False

    if args.subset_by_detector and args.subset_quantile:
        print('args.subset_by_detector', args.subset_by_detector)
        df_subset, det_col, det_dir, keep_ratio = _pick_subset_by_detector(
            df_full, detectors, args.subset_by_detector, args.subset_quantile, take=args.subset_take
        )
        subset_used = True

        # ---- Subset evaluation (write to existing output paths; use user-provided subset output paths if available, otherwise skip saving) ----
        if args.output_summary_json_subset or args.output_plot_subset:
            df_scores_to_eval = df_subset
            def _evaluate_and_save(df_input, out_json, out_png, title_suffix="", detectors=None):
                final_evaluation = {}
                for detector in detectors or []:
                    method_name = detector.get_name()
                    direction = detector.get_direction()
                    if method_name not in df_input.columns:
                        continue
                    df_m = df_input.dropna(subset=[method_name])
                    if len(np.unique(df_m['ground_truth_label'])) < 2:
                        continue
                    y_true = df_m['ground_truth_label'].values
                    y_scores = df_m[method_name].values * direction
                    perf = evaluate_performance_common(y_true, y_scores)
                    final_evaluation[method_name] = {"overall_performance": perf}

                if out_json:
                    with open(out_json, 'w') as f:
                        json.dump(final_evaluation, f, indent=2)

            _evaluate_and_save(
                df_subset,
                args.output_summary_json_subset,
                args.output_plot_subset,
                title_suffix=f"[{args.subset_by_detector}-{args.subset_take}-q={args.subset_quantile}]",
                detectors=detectors, 
            )
    else:
        keep_ratio = 1.0  # No subset, use default ratio for random control

    # ---- Random control evaluation (if enabled) ----
    if args.with_random_control:
        df_rand = _random_subset(df_full, keep_ratio, seed=args.random_seed)
        random_used = True

        # Auto-complete default output paths (if not provided)
        out_json_r = args.output_summary_json_random
        out_png_r  = args.output_plot_random
        if not out_json_r and args.output_summary_json_subset:
            base, ext = os.path.splitext(args.output_summary_json_subset)
            out_json_r = f"{base}_RANDOM{ext or '.json'}"
        if not out_png_r and args.output_plot_subset:
            base, ext = os.path.splitext(args.output_plot_subset)
            out_png_r = f"{base}_RANDOM{ext or '.png'}"

        # Evaluate random subset
        def _evaluate_and_save(df_input, out_json, out_png, title_suffix="", detectors=None):
            final_evaluation = {}
            for detector in detectors or []:
                method_name = detector.get_name()
                direction = detector.get_direction()
                if method_name not in df_input.columns:
                    continue
                df_m = df_input.dropna(subset=[method_name])
                if len(np.unique(df_m['ground_truth_label'])) < 2:
                    continue
                y_true = df_m['ground_truth_label'].values
                y_scores = df_m[method_name].values * direction
                perf = evaluate_performance_common(y_true, y_scores)
                final_evaluation[method_name] = {"overall_performance": perf}

            if out_json:
                with open(out_json, 'w') as f:
                    json.dump(final_evaluation, f, indent=2)

        _evaluate_and_save(
            df_rand,
            out_json_r,
            out_png_r,
            title_suffix=f"[random keep≈{keep_ratio:.2f}, seed={args.random_seed}]",
            detectors=detectors,
        )

if __name__ == "__main__":
    main()