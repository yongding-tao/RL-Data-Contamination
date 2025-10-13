from .base_detector import BaseDetector
import numpy as np

class PPLDetector(BaseDetector):
    def get_name(self):
        return "perplexity_score"

    def get_direction(self):
        return -1 # Lower PPL is more suspicious

    def _compute_ppl(self, logprobs):
        if not logprobs:
            return np.nan
        return np.exp(-np.mean(logprobs))

    def calculate_score(self, data_item):
        greedy_result = data_item.get('original_greedy_results', [{}])[0]
        logprobs = [lp for lp in greedy_result.get('logprobs', []) if lp is not None]
        ppl_original = self._compute_ppl(logprobs)

        return ppl_original