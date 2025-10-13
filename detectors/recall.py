from .base_detector import BaseDetector
import numpy as np

class RecallDetector(BaseDetector):
    def get_name(self):
        return "recall_score"
    
    def get_direction(self):
        return 1 # Lower score is more likely to be contamination

    def calculate_score(self, data_item):
        """
        Calculate RECALL score, adapted for our differential detection data.
        Score = LL(perturbed_response) / LL(original_response)
        """
        # original_results: [Greedy(original prompt)]
        # perturbed_results: [Greedy(perturbed prompt)]
        
        original_result = data_item.get('original_greedy_results', [{}])[0]
        perturbed_result = data_item.get('perturbed_greedy_results', [{}])[0]
        
        # Extract average log-likelihood (Log-Likelihood) for each response
        original_logprobs = [lp for lp in original_result.get('logprobs', []) if lp is not None]
        perturbed_logprobs = [lp for lp in perturbed_result.get('logprobs', []) if lp is not None]
        
        if not original_logprobs or not perturbed_logprobs:
            return np.nan

        # Calculate average log-likelihood
        ll_original = np.mean(original_logprobs)
        # print('ll_original:', ll_original)
        ll_perturbed = np.mean(perturbed_logprobs)
        # print('ll_perturbed:', ll_perturbed)
        
        # Avoid division by zero
        if ll_original == 0:
            return np.nan
        
        ll_change = ll_perturbed / ll_original
        return ll_change