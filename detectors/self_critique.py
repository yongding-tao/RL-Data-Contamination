from .base_detector import BaseDetector
import numpy as np

def calculate_penalized_cosine_similarity(list1, list2):
    if not list1 or not list2: return 0.0
    len1, len2 = len(list1), len(list2)
    max_len = max(len1, len2)
    padded_arr1, padded_arr2 = np.zeros(max_len), np.zeros(max_len)
    padded_arr1[:len1], padded_arr2[:len2] = list1, list2
    dot_product = np.dot(padded_arr1, padded_arr2)
    norm_a, norm_b = np.linalg.norm(padded_arr1), np.linalg.norm(padded_arr2)
    if norm_a == 0 or norm_b == 0: return 0.0
    cosine_sim = dot_product / (norm_a * norm_b)
    length_penalty = min(len1, len2) / max(len1, len2)
    return cosine_sim * length_penalty

class SelfCritiqueDetector(BaseDetector):
    def get_name(self):
        return "self_critique_score"
    
    def get_direction(self):
        return 1 # High similarity = contamination

    def calculate_score(self, data_item):
        if 'original_greedy_results' not in data_item or 'critique_greedy_results' not in data_item:
            return np.nan
            
        first_pass_result = data_item['original_greedy_results'][0]
        second_pass_result = data_item['critique_greedy_results'][0]
        
        first_pass_entropies = first_pass_result.get('entropies', [])
        second_pass_entropies = second_pass_result.get('entropies', [])
        
        if not first_pass_entropies or not second_pass_entropies:
            return np.nan
            
        return calculate_penalized_cosine_similarity(first_pass_entropies, second_pass_entropies)