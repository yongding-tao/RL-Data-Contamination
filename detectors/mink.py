from .base_detector import BaseDetector
import numpy as np

class MinkDetector(BaseDetector):
    def __init__(self, mink_ratio=0.2, use_plus_plus=False):
        super().__init__()
        self.mink_ratio = mink_ratio
        self.use_plus_plus = use_plus_plus
        self.name = f"mink_{'plus_plus_' if use_plus_plus else ''}{self.mink_ratio*100:.0f}p_score"

    def get_name(self):
        return self.name
        
    def get_direction(self):
        return 1 # Higher score is more suspicious

    def calculate_score(self, data_item):
        greedy_result = data_item.get('original_greedy_results', [{}])[0]
        actual_logprobs = [lp for lp in greedy_result.get('logprobs', []) if lp is not None]
        
        if not actual_logprobs: return np.nan
        
        k_length = int(len(actual_logprobs) * self.mink_ratio)
        if k_length == 0: return np.nan

        if not self.use_plus_plus:
            # Min-K% Prob (logic unchanged)
            topk_logprobs = np.sort(actual_logprobs)[:k_length]
            return -np.mean(topk_logprobs)
        else:
            # <<< directly read pre-computed mu and sigma from data >>>
            mus = greedy_result.get('mus', [])
            sigmas = greedy_result.get('sigmas', [])
            
            if not mus or not sigmas: 
                # If generation stage failed to calculate for some reason, return NaN
                return np.nan

            token_log_probs = np.array(actual_logprobs)
            min_len = min(len(token_log_probs), len(mus))
            
            # Ensure sufficient data for slicing and calculation
            if min_len < k_length: return np.nan
            
            # Use pre-computed values
            mink_plus_scores = (token_log_probs[:min_len] - np.array(mus)[:min_len]) / np.array(sigmas)[:min_len]
            
            topk_plus = np.sort(mink_plus_scores)[:k_length]
            return np.mean(topk_plus)