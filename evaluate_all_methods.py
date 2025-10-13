import os
import json
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score

from detectors.dime import DIMEDetector
from detectors.dime_temp import DimeTempDetector
from detectors.mink import MinkDetector
from detectors.ppl import PPLDetector
from detectors.cdd import CDDDetector
from detectors.recall import RecallDetector
from detectors.self_critique import SelfCritiqueDetector
from detectors.self_critique_ablation import SelfCritiqueAblationDetector

FPR = []
TPR = []

def evaluate_performance_pop(y_true, y_scores):
    """
    A helper function to calculate a complete set of evaluation metrics,
    including AUC, optimal F1, Youden's J, TPR at fixed FPR,
    and F1 and Accuracy corresponding to Youden's J threshold.
    """
    # --- 0. Data preprocessing ---
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
    
    # --- 1. Calculate basic ROC curve and AUC ---
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    FPR.append(fpr.tolist())
    TPR.append(tpr.tolist())
    roc_auc = auc(fpr, tpr)
    

    # --- 2. Calculate optimal F1 score (F1-based threshold) ---
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    fscore = (2 * precision * recall) / (precision + recall + 1e-6)
    best_f1_idx = np.argmax(fscore[:-1]) if len(fscore) > 1 else 0
    optimal_threshold_f1 = pr_thresholds[best_f1_idx]
    best_f1 = fscore[best_f1_idx]
    y_pred_f1 = (y_scores >= optimal_threshold_f1).astype(int)
    accuracy_at_best_f1 = accuracy_score(y_true, y_pred_f1)
    
    # --- 3. Calculate Youden's J Statistic ---
    youden_j_scores = tpr - fpr
    best_youden_idx = np.argmax(youden_j_scores)
    youden_j_score = youden_j_scores[best_youden_idx]
    optimal_threshold_youden = roc_thresholds[best_youden_idx]

    # Use Youden threshold for prediction
    y_pred_youden = (y_scores >= optimal_threshold_youden).astype(int)
    
    # Calculate corresponding F1 score
    tp_youden = np.sum((y_true == 1) & (y_pred_youden == 1))
    fp_youden = np.sum((y_true == 0) & (y_pred_youden == 1))
    fn_youden = np.sum((y_true == 1) & (y_pred_youden == 0))
    
    precision_youden = tp_youden / (tp_youden + fp_youden + 1e-6)
    recall_youden = tp_youden / (tp_youden + fn_youden + 1e-6)
    
    f1_at_youden_threshold = (2 * precision_youden * recall_youden) / (precision_youden + recall_youden + 1e-6)
    
    # Calculate corresponding accuracy
    accuracy_at_youden_threshold = accuracy_score(y_true, y_pred_youden)

    # --- 4. Calculate TPR at fixed FPR (unchanged) ---
    target_fpr = 0.05
    indices_above_target = np.where(fpr >= target_fpr)[0]
    if len(indices_above_target) > 0:
        target_idx = indices_above_target[0] - 1 if indices_above_target[0] > 0 else 0
        tpr_at_fpr_5 = tpr[target_idx]
    else:
        tpr_at_fpr_5 = tpr[-1] if len(tpr) > 0 else np.nan

    # --- 5. Return all metrics ---
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
    

def evaluate_performance(y_true, y_scores):
    """
    A helper function to calculate a complete set of evaluation metrics,
    including AUC, optimal F1, Youden's J, TPR at fixed FPR,
    and F1 and Accuracy corresponding to Youden's J threshold.
    """
    # --- 0. Data preprocessing ---
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
    
    # --- 1. Calculate basic ROC curve and AUC ---
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # --- 2. Calculate optimal F1 score (F1-based threshold) ---
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    fscore = (2 * precision * recall) / (precision + recall + 1e-6)
    best_f1_idx = np.argmax(fscore[:-1]) if len(fscore) > 1 else 0
    optimal_threshold_f1 = pr_thresholds[best_f1_idx]
    best_f1 = fscore[best_f1_idx]
    y_pred_f1 = (y_scores >= optimal_threshold_f1).astype(int)
    accuracy_at_best_f1 = accuracy_score(y_true, y_pred_f1)
    
    # --- 3. Calculate Youden's J Statistic ---
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
    
    # 3. Calculate corresponding accuracy
    accuracy_at_youden_threshold = accuracy_score(y_true, y_pred_youden)

    # --- 4. Calculate TPR at fixed FPR (unchanged) ---
    target_fpr = 0.05
    indices_above_target = np.where(fpr >= target_fpr)[0]
    if len(indices_above_target) > 0:
        target_idx = indices_above_target[0] - 1 if indices_above_target[0] > 0 else 0
        tpr_at_fpr_5 = tpr[target_idx]
    else:
        tpr_at_fpr_5 = tpr[-1] if len(tpr) > 0 else np.nan

    # --- 5. Return all metrics ---
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

def main():
    parser = argparse.ArgumentParser(description="Calculate performance for all modular detection methods.")
    parser.add_argument("--input_file", type=str, required=True, help="JSONL file generated by generate_full_data.py.")
    parser.add_argument("--output_summary_json", type=str, required=True, help="JSON filename to save performance comparison of all methods.")
    parser.add_argument("--output_plot", type=str, required=True, help="Image name to save DIME method performance analysis plot.")
    parser.add_argument("--mink_ratio", type=float, default=0.2, help="Percentage k for Min-K% method. The default setting in original paper")
    parser.add_argument("--output_summary_json_subset", type=str, default=None,
                        help="JSON output path for subset evaluation.")
    parser.add_argument("--output_plot_subset", type=str, default=None,
                        help="ROC plot output path for subset evaluation.")   
    args = parser.parse_args()

    # --- 1. Instantiate all detectors to run ---
    print("Initializing all detectors...")
    detectors = [
        PPLDetector(),
        MinkDetector(mink_ratio=args.mink_ratio, use_plus_plus=False), # Min-K%
        MinkDetector(mink_ratio=args.mink_ratio, use_plus_plus=True),  # Min-K%++
        RecallDetector(),
        CDDDetector(alpha=0.05), # The default setting in original paper
        DimeTempDetector(),
        DIMEDetector(),
        SelfCritiqueDetector(),
    ]
    
    # --- 2. Read data and calculate all scores ---
    print(f"Reading data from {args.input_file} and calculating all scores...")
    all_scores_list = []
    with open(args.input_file, 'r') as f:
        for line in tqdm(f, desc="Processing samples"):
            data_item = json.loads(line)
            scores = {
                "ground_truth_label": data_item['ground_truth_label'],
                "data_source": data_item.get('data_source', 'unknown'),
                "original_user_content": data_item.get('original_user_content')
            }
            for detector in detectors:
                scores[detector.get_name()] = detector.calculate_score(data_item)
            all_scores_list.append(scores)
            
    df_scores = pd.DataFrame(all_scores_list)

    # --- 3. Evaluate all methods ---
    final_evaluation = {}
    print("\nCalculating and saving evaluation results...")
    for detector in detectors:
        method_name = detector.get_name()
        direction = detector.get_direction()
        
        df_method = df_scores.dropna(subset=[method_name])
        y_true = df_method['ground_truth_label'].values
        y_scores = df_method[method_name].values * direction
        
        overall_perf = evaluate_performance_pop(y_true, y_scores)
        
        breakdown = {}
        mean_auc = 0
        for source, group in df_method.groupby('data_source'):
            group_perf = evaluate_performance(group['ground_truth_label'].values, group[method_name].values * direction)
            breakdown[source] = group_perf
            if not np.isnan(group_perf['roc_auc']):
                mean_auc += group_perf['roc_auc']

        mean_auc /= len(breakdown) if breakdown else 1    

        final_evaluation[method_name] = {
            "overall_performance": overall_perf,
            "mean_auc": mean_auc,
            "breakdown_by_source": breakdown
        }

    with open(args.output_summary_json, 'w') as f:
        json.dump(final_evaluation, f, indent=4)
    print(f"\nEvaluation summary of all methods saved to: {args.output_summary_json}")

if __name__ == '__main__':
    main()
