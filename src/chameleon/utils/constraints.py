import Levenshtein
from textattack.constraints import Constraint

class Levenshtein_length_Constraint(Constraint):
    """Ensure the transformed text meets Levenshtein distance and length constraints."""

    def __init__(self, original_sentence, min_distance, min_length, max_length):
        super().__init__(compare_against_original=True)
        self.original = original_sentence
        self.min_distance = min_distance
        self.min_length = min_length
        self.max_length = max_length

    def _check_constraint(self, transformed_text, reference_text):
        candidate = transformed_text.text
        if not (self.min_length <= len(candidate) <= self.max_length):
            return False
        return Levenshtein.distance(self.original, candidate) >= self.min_distance