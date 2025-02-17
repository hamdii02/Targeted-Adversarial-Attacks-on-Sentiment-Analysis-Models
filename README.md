# Chameleon: Text Classification Model Analysis

This is a Python framework designed to analyze text classification models by finding sentences with similar sentiment scores to a given target sentence. It supports HuggingFace text classification models and provides tools for probing and generating adversarial examples.

Features
    Model Support: Works with HuggingFace text classification models.
    Probes: Includes probes to analyze models and find sentences with similar sentiment scores.

    Modular Design: Clean and modular structure for easy extension.

# Available Probes:  TextCraft


**TextCraft** is a Python library designed to analyze and perturb text classification models, specifically focusing on generating adversarial examples that retain similar sentiment scores to the original sentence. By using state-of-the-art techniques such as **paraphrasing**, **word swapping**, and **Particle Swarm Optimization (PSO)**, it creates modified sentences that challenge the model while preserving the intended classification probabilities. This is done in a black-box setting where only the output of the model is available.

## Features

- **Model Support**: Works with HuggingFace text classification models.
- **Adversarial Attack Generation**: Uses PSO and large models for mask infilling and word embedding swaps to generate adversarial examples.
- **Text Paraphrasing**: Leverages BART to create semantically similar paraphrases with the same sentiment classification.
- **Black-box Attack**: Operates in a black-box scenario, where only the output of the model is available.
- **Grammar and Syntax Preservation**: It generated adversarial examples maintain good grammar and acceptable syntax.
- **Modular Design**: Clean and modular structure for easy extension.

## Available Probes: 

- **Goal**: Generate paraphrased sentences that match the sentiment classification probabilities of a given sentence.
- **Constraints**: Applies Levenshtein distance and length constraints while ensuring that the paraphrases adhere to the target sentiment.
- **Attack Process**: First, generate paraphrases, then perturb them using PSO and word embedding swaps to match the classification output of the original 
sentence.


# Results


| Sentence | Positive score | Neutral score | Negative score |
| --- | --- | --- | --- |
| My grandmother's secret sauce is the best ever made! | 0.9619 | 0.0256 | 0.0123 |
| I feel it should be a positive thing for us to look | 0.9615 | 0.0262 | 0.0121 |
| I have this excellent secret love. Her sauce is the best? | 0.9624 | 0.0259 | 0.0117 |
| good science is great as a finding of my good brain salsa? | 0.9616 | 0.0264 | 0.0120 |


Example Output


Found sentence: I have this excellent secret love. Her sauce is the best?
Scores: {'positive': 0.95, 'neutral': 0.03, 'negative': 0.02}
Elapsed time: 12.34

Project Structure

```
src/  
│── chameleon/  
│   ├── models/              # Handles model loading and processing  
│   │   ├── __init__.py  
│   │   ├── base.py          # Base class for models  
│   │   ├── huggingface.py   # Hugging Face model implementation  
│   │  
│   ├── probes/              # Probe implementations  
│   │   ├── __init__.py  
│   │   ├── score_matching_probe.py   # Generates sentences with similar classification scores  
│   │  
│   ├── tests/               # Unit tests  
│   │   ├── __init__.py  
│   │   ├── test_huggingface_model.py  # Tests for Hugging Face model  
│  
│── utils/                   # Utility functions and classes  
│   ├── constraints.py        # Custom constraints for TextAttack  
│   ├── goal_functions.py     # Custom goal functions for TextAttack  
│   ├── attack_utils.py       # Helper functions for attacks  
│   ├── paraphraser.py        # Paraphrases sentences under constraints  
│  
│── .gitignore                # Files to ignore in version control  
│── README.md                 # Main project documentation  

```
