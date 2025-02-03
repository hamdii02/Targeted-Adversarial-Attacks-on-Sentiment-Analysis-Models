import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def compute_target_scores_and_label(sentence, model, tokenizer, n_decimal=3):
    """Compute the original sentiment scores and extract the target label."""
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    scores = torch.softmax(logits, dim=1).numpy()[0]
    rounded_scores = np.round(scores, decimals=n_decimal)
    label = np.argmax(rounded_scores)
    return rounded_scores, label