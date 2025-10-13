from .base_detector import BaseDetector
import numpy as np

def calculate_penalized_cosine_similarity(list1, list2):
    """Calculate cosine similarity with length difference penalty."""
    if not list1 or not list2: 
        return 0.0
    
    len1, len2 = len(list1), len(list2)
    max_len = max(len1, len2)
    
    padded_arr1 = np.zeros(max_len)
    padded_arr2 = np.zeros(max_len)
    padded_arr1[:len1] = list1
    padded_arr2[:len2] = list2
    
    dot_product = np.dot(padded_arr1, padded_arr2)
    norm_a = np.linalg.norm(padded_arr1)
    norm_b = np.linalg.norm(padded_arr2)
    
    if norm_a == 0 or norm_b == 0: 
        return 0.0
        
    cosine_sim = dot_product / (norm_a * norm_b)
    length_penalty = min(len1, len2) / max(len1, len2)
    
    return cosine_sim * length_penalty


class DimeTempDetector(BaseDetector):
    """
    Dime-temp method: Compare entropy sequence consistency between Greedy and Random samples under original prompt.
    This is an internal consistency detector.
    """
    def get_name(self):
        return "dime_temp_score"
    
    def get_direction(self):
        # In this original method, we assume high similarity/high stability = contamination
        return 1 

    def calculate_score(self, data_item):
        """
        Calculate Dime-temp contamination score.
        Score = Avg_Similarity * Stability
        """
        # The data used by this method is:
        # original_greedy_results: [Greedy(original prompt)]
        # original_random_results: [Random_1(original prompt), Random_2, ...]
        
        greedy_results = data_item.get('original_greedy_results', [])
        random_results = data_item.get('original_random_results', [])
        
        if not greedy_results or not random_results:
            return np.nan
            
        greedy_entropies = greedy_results[0].get('entropies', [])
        if not greedy_entropies:
            return np.nan

        # 1. Calculate entropy sequence similarity between Greedy sample and each Random sample
        similarities = []
        for r_res in random_results:
            random_entropies = r_res.get('entropies')
            if random_entropies:
                sim = calculate_penalized_cosine_similarity(greedy_entropies, random_entropies)
                similarities.append(sim)

        if not similarities:
            return np.nan

        # 2. Calculate average similarity score
        avg_similarity_score = np.mean(similarities)
        
        # 3. Calculate stability score
        std_dev_similarity = np.std(similarities)
        stability_score = 1 - min(std_dev_similarity, 1.0) 

        # 4. Final, training-free discrimination score
        contamination_score = avg_similarity_score * stability_score
        
        return contamination_score