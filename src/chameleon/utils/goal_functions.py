import numpy as np
import torch
from textattack.goal_functions import GoalFunction
from textattack.goal_function_results import ClassificationGoalFunctionResult

class RoundedScoreGoal_prob(GoalFunction):
    """Custom goal function to match target sentiment scores."""

    def __init__(self, model, target_scores, label_order, n_decimal=4):
        super().__init__(model)
        self.target_scores = np.round(np.array(target_scores), decimals=n_decimal)
        self.label_order = label_order
        self.n_decimal = n_decimal
        self._validate_target_scores()

    def _validate_target_scores(self):
        """Ensure rounded targets sum to ~1 and are valid probabilities."""
        if not np.isclose(self.target_scores.sum(), 1.0, atol=1e-2):
            raise ValueError("Rounded target scores must sum to ~1")
        if (self.target_scores < 0).any() or (self.target_scores > 1).any():
            raise ValueError("All target scores must be between 0 and 1")

    def _is_goal_complete(self, model_output, attacked_text):
        """Check if the generated text meets the target probabilities."""
        scores = model_output.numpy().flatten()
        print(f"obtained Scores: {scores} | targer Scores {self.target_scores}")

        found_valid_example = False

        # Loop through each decimal level from 1 to n_decimal
        for decimals in range(1, self.n_decimal + 1):
            rounded_scores = np.round(scores, decimals=decimals)
            target_scores = np.round(np.array(self.target_scores), decimals=decimals)
            is_close = np.allclose(rounded_scores, target_scores)

            if is_close:
                print(f"Match found at {decimals} decimal places:")
                print(f"Sentence: '{attacked_text.text}'")
                print(f"Rounded Scores: {rounded_scores}")
                found_valid_example = True

        if not found_valid_example:
            print("No match found for any decimal precision up to n_decimal.")
        rounded_scores_final = np.round(scores, decimals=self.n_decimal)
        return np.allclose(rounded_scores_final, self.target_scores)

    def _get_score(self, model_output, attacked_text):
        """Compute score as the negative distance from the target probabilities."""
        scores = model_output.numpy().flatten()
        rounded_scores = np.round(scores, decimals=self.n_decimal)
        return -np.linalg.norm(rounded_scores - self.target_scores)

    def _process_model_outputs(self, inputs, model_outputs):
        """Process model outputs to get probabilities."""
        probabilities = torch.nn.functional.softmax(model_outputs, dim=-1)
        return probabilities

    def _goal_function_result_type(self):
        return ClassificationGoalFunctionResult