from abc import ABC, abstractmethod

class BaseDetector(ABC):
    """
    Abstract base class for all detection methods.
    """
    def __init__(self, **kwargs):
        """
        Initialize detector. Can preload models, etc.
        """
        pass

    @abstractmethod
    def get_name(self):
        """Return the name of this detector."""
        pass

    @abstractmethod
    def calculate_score(self, data_item):
        """
        Calculate contamination score for a single data item.
        
        Args:
            data_item (dict): A line of data read from JSONL file.

        Returns:
            float: Contamination score.
        """
        pass

    def get_direction(self):
        """
        Return whether higher scores are more suspicious.
        1: Higher is more suspicious
        -1: Lower is more suspicious
        """
        return 1 # Default: higher is more suspicious