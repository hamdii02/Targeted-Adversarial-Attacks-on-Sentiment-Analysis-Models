from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .base import BaseModel


class HuggingFaceModel(BaseModel):
    """A wrapper for HuggingFace models."""

    def __init__(self, model: str):
        """Initialize the model.

        Parameters
        ----------
        model : str
            The name of the HuggingFace model to use. For example,
            "distilbert-base-uncased-finetuned-sst-2-english".
            Must be a text classification model.
        """
        self._model_name = model
        self._pipe = pipeline("text-classification", model=self._model_name, top_k=None)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)

    def predict(self, sentence):
        return {el["label"]: el["score"] for el in self._pipe(sentence)[0]}

    def labels(self):
        return list(self._pipe.model.config.id2label.values())
