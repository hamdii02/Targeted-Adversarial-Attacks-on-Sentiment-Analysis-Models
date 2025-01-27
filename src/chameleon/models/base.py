from typing import Dict, List
import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """This class defines the abstract interface for all model wrappers."""

    @abstractmethod
    def predict(self, sentence: str) -> Dict[str, float]:
        """Predict the sentiment of a sentence.

        Parameters
        ----------
        sentence : str
            The sentence to predict the sentiment of.

        Returns
        -------
        Dict[str, float]
            A dictionary with the sentiment predictions of the sentence, in the
            form of a mapping from sentiment labels to their corresponding
            scores/probabilities. For example::

                {
                    "positive": 0.9,
                    "negative": 0.1,
                    "neutral": 0.0
                }

            Labels may vary depending on the model.
        """
        ...
    
    @abstractmethod
    def labels(self) -> List[str]:
        """Return the labels supported by the model.

        Returns
        -------
        List[str]
            A list of the labels supported by the model. For example:
            ``["positive", "negative", "neutral"]``.
        """
        ...
