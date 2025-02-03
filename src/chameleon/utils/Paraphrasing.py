import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import Levenshtein


def generate_best_paraphrase(input_sentence: str,
                             min_levenshtein_distance: int = 30,
                             min_length: int = 30,
                             max_length: int = 60,
                             num_return_sequences: int = 20,
                             temperature: float = 1.5,
                             top_k: int = 50,
                             top_p: float = 0.95,
                             cosine_similarity_threshold: float = 0.8,
                             threshold_attempts = 300) -> str:
    """
    Generates diverse paraphrases for a given input sentence and returns the best one
    based on cosine similarity and constraints. Keeps generating paraphrases
    until one has a cosine similarity score greater than the threshold.

    Args:
        input_sentence (str): Sentence to paraphrase.
        min_levenshtein_distance (int): Minimum Levenshtein distance from input_sentence.
        min_length (int): Minimum length constraint for generated paraphrases.
        max_length (int): Maximum length constraint for generated paraphrases.
        num_return_sequences (int): Number of paraphrases to generate per attempt.
        temperature (float): Temperature parameter for diversity during sampling.
        top_k (int): Top-k sampling parameter.
        top_p (float): Nucleus sampling (top-p) parameter.
        cosine_similarity_threshold (float): Minimum cosine similarity score for valid paraphrase.

    Returns:
        str: Best paraphrase based on cosine similarity or message indicating failure.
    """
    
    # Load BART paraphrase model and tokenizer
    bart_model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
    bart_tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')
    # Define device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bart_model = bart_model.to(device)

    # Load DistilBERT model for embeddings
    embedding_model = DistilBertModel.from_pretrained('lxyuan/distilbert-base-multilingual-cased-sentiments-student')
    embedding_tokenizer = DistilBertTokenizer.from_pretrained('lxyuan/distilbert-base-multilingual-cased-sentiments-student')
    embedding_model = embedding_model.to(device)

    # Tokenize the input sentence
    batch = bart_tokenizer(input_sentence, return_tensors='pt').to(device)


    attempts = 0
    # Loop to keep generating paraphrases until similarity score > 0.8
    while attempts < threshold_attempts:
        # Generate diverse paraphrases
        generated_ids = bart_model.generate(
            batch['input_ids'],
            num_return_sequences=num_return_sequences,
            num_beams=1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.5,
        )
        paraphrases = bart_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Filter based on Levenshtein distance and length
        valid_paraphrases = [
            p for p in paraphrases if
            Levenshtein.distance(input_sentence, p) >= min_levenshtein_distance and
            min_length <= len(p) <= max_length
        ]

        # If valid paraphrases are found, proceed to similarity check
        if valid_paraphrases:
            # Function to compute embeddings
            def compute_embedding(sentence):
                inputs = embedding_tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(device)
                with torch.no_grad():
                    outputs = embedding_model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

            # Compute embeddings for the original sentence and valid paraphrases
            original_embedding = compute_embedding(input_sentence)
            paraphrase_embeddings = [compute_embedding(p) for p in valid_paraphrases]

            # Compute cosine similarity
            similarities = [cosine_similarity([original_embedding], [embedding])[0][0] for embedding in paraphrase_embeddings]

            # Check if any paraphrase meets the similarity threshold
            for i, similarity in enumerate(similarities):
                if similarity > cosine_similarity_threshold:
                    best_paraphrase = valid_paraphrases[i]
                    return best_paraphrase

        # If no valid paraphrase is found, continue the loop until success

    return "No valid paraphrase found after several attempts."

