from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

from ..models.base import BaseModel


@dataclass
class ProbeResult:
    """The result of a probe run, returning the matching sentence and scores."""

    sentence: str
    scores: Dict[str, float]


class BaseProbe(ABC):
    """This class defines the abstract interface for all probes.

    Parameters
    ----------
    model : BaseModel
        A wrapper of the model to probe.
    target : str
        The target sentence which sentiment scores should be matched.
    """

    def __init__(self, model: BaseModel, target: str):
        self.model = model
        self.target = target

    @abstractmethod
    def run(self, **kwargs) -> ProbeResult:
        """Run the probe.

        Finds the sentence that matches the sentiment score of the target
        sentence.
        """
        ...
