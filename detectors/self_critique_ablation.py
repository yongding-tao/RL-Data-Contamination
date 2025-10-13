from .base_detector import BaseDetector
import numpy as np

def _penalized_cosine(list1, list2):
    if not list1 or not list2:
        return 0.0
    len1, len2 = len(list1), len(list2)
    m = max(len1, len2)
    a = np.zeros(m, dtype=float)
    b = np.zeros(m, dtype=float)
    a[:len1] = list1
    b[:len2] = list2
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    cos = float(np.dot(a, b) / (na * nb))
    length_penalty = min(len1, len2) / max(len1, len2)
    return cos * length_penalty

class SelfCritiqueAblationDetector(BaseDetector):
    """
    Ablation version of Self-Critique:
    Second generation does not concatenate the first answer, only appends "Please answer in an unfamiliar/unusual way" after the original question.
    Score = Penalized cosine similarity of token-level entropy lists from two generations.
    """
    def get_name(self):
        return "self_critique_ablation_score"

    def get_direction(self):
        return 1  # Higher similarity -> more like memory replay -> higher contamination possibility

    def calculate_score(self, data_item):
        if 'original_greedy_results' not in data_item or 'unfamiliar_greedy_results' not in data_item:
            return np.nan

        first = data_item['original_greedy_results'][0]
        second = data_item['unfamiliar_greedy_results'][0]

        ent1 = first.get('entropies', []) or []
        ent2 = second.get('entropies', []) or []
        if not ent1 or not ent2:
            return np.nan

        return _penalized_cosine(ent1, ent2)