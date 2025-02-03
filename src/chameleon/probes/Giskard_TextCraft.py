from textattack import Attack, AttackArgs
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset
from textattack.transformations import (
    WordSwapRandomCharacterDeletion, CompositeTransformation,
    WordSwapEmbedding, WordSwapMaskedLM
)

from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.search_methods import ParticleSwarmOptimization
from ..utils.goal_functions import RoundedScoreGoal_prob
from ..utils.constraints import Levenshtein_length_Constraint
from ..utils.Paraphrasing import generate_best_paraphrase
from ..models.base import BaseModel,HuggingFaceModel
from probes.base import base
from ..utils.attack_utils import compute_target_scores_and_label
from dataclasses import dataclass
from typing import Dict
import time
from math import log10

@dataclass
class TextAttackProbeResult:
    """The result of a TextAttack probe run."""
    sentence: str
    scores: Dict[str, float]
    elapsed_time: float

class TextAttackProbe(base):
    """A probe that uses TextAttack to generate adversarial examples."""

    def __init__(self, model: HuggingFaceModel, target_sentence: str, min_levenshtein: int = 30, min_length: int = 40, max_length: int = 60):
        self.model = model
        self.target_sentence = target_sentence
        self.min_levenshtein = min_levenshtein
        self.min_length = min_length
        self.max_length = max_length

    def run(self, epsilon: float = 1e-3) -> TextAttackProbeResult:
        """Run the probe to generate an adversarial example."""

        # Wrap the model for TextAttack
        wrapped_model = HuggingFaceModelWrapper(self.model._model, self.model._tokenizer)

        # transformations that will be applied on the given sentence
        transformation = CompositeTransformation([
            WordSwapEmbedding(max_candidates=30),
            WordSwapMaskedLM(method="bae", max_candidates=40),
            WordSwapRandomCharacterDeletion()
        ])

        # Constraints to be respected by the transformations
        constraints = [
            RepeatModification(),
            StopwordModification(),
            Levenshtein_length_Constraint(self.target_sentence, self.min_levenshtein, self.min_length, self.max_length)

        ]

        # Define search method
        search_method = ParticleSwarmOptimization(pop_size=80, max_iters=40, post_turn_check=True, max_turn_retries=10)

        # Compute target scores and label
        target_scores, label = compute_target_scores_and_label(self.target_sentence, self.model._model, self.model._tokenizer)

        # Define goal function
        goal_function = RoundedScoreGoal_prob(
            model=wrapped_model,
            target_scores=target_scores,
            label_order=self.model.labels(),
            n_decimal = int(-log10(epsilon))
        )

        # Define attack
        text_manipulator = Attack(goal_function, constraints, transformation, search_method)

        generated_sentence = generate_best_paraphrase(self.target_sentence)

        # Run attack
        start_time = time.time()
        result = text_manipulator.attack(generated_sentence, label)       
        elapsed_time = time.time() - start_time


        # Extract results
        adversarial_sentence = result.perturbed_text()
        adversarial_scores = self.model.predict(adversarial_sentence)


        return TextAttackProbeResult(
            sentence=adversarial_sentence,
            scores=adversarial_scores,
            elapsed_time=elapsed_time
        )