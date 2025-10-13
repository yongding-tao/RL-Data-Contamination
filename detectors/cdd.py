from .base_detector import BaseDetector
import numpy as np

def levenshtein_distance_token(tokens1, tokens2):
    if len(tokens1) > len(tokens2):
        tokens1, tokens2 = tokens2, tokens1

    token_to_detect = 100
    if len(tokens1) > token_to_detect: # Limit maximum calculation length to prevent slow computation for long texts
        tokens1 = tokens1[:token_to_detect]
        tokens2 = tokens2[:token_to_detect]

    distances = range(len(tokens1) + 1)
    for index2, token2 in enumerate(tokens2):
        new_distances = [index2 + 1]
        for index1, token1 in enumerate(tokens1):
            if token1 == token2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances
    return distances[-1]

class CDDDetector(BaseDetector):
    def __init__(self, alpha=0.05, xi=0.01):
        super().__init__()
        self.alpha = alpha
        self.xi = xi # Note: xi is the threshold for final classification, in our framework we only calculate scores
        self.name = f"cdd_peak_score_alpha_{self.alpha}"

    def get_name(self):
        return self.name
    
    def get_direction(self):
        return 1 # Higher Peak score is more suspicious

    def calculate_score(self, data_item):
        greedy_results = data_item.get('original_greedy_results', [])
        random_results = data_item.get('original_random_results', [])
        
        if not greedy_results or not random_results:
            return np.nan
            
        greedy_result = greedy_results[0]
        greedy_token_ids = greedy_result.get('token_ids')
        if not greedy_token_ids:
            return np.nan

        # 1. Calculate token-level edit distance between Greedy sample and each Random sample
        edit_distances = []
        max_len = len(greedy_token_ids)
        for r_res in random_results:
            random_token_ids = r_res.get('token_ids')
            if not random_token_ids:
                continue
            
            dist = levenshtein_distance_token(greedy_token_ids, random_token_ids)
            edit_distances.append(dist)
            max_len = max(max_len, len(random_token_ids))

        if not edit_distances:
            return np.nan

        # 2. Calculate Peak score (proportion of samples with low edit distance)
        # threshold = self.alpha * max_len # This is the definition in the CDD paper
        threshold = self.alpha * np.mean([len(r.get('token_ids',[])) for r in random_results] + [len(greedy_token_ids)])
        
        count_below_threshold = sum(1 for d in edit_distances if d <= threshold)
        total_samples = len(edit_distances)
        
        peak_score = count_below_threshold / total_samples if total_samples > 0 else 0
        return peak_score