from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import bert_score


# Example Reference and Hypothesis Lists
references = [["the", "cat", "is", "on", "the", "mat"]]  # Reference should be a list of lists
hypothesis = ["the", "cat", "is", "on", "mat"]  # Hypothesis is a single list

# Compute BLEU-4 Score
bleu_score = sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25),
                           smoothing_function=SmoothingFunction().method1)

# P, R, F1 = bert_score.score(hypothesis, references, lang="en", model_type="microsoft/deberta-xlarge-mnli")


print("BLEU-4 Score:", bleu_score)
print("BERT F1-score:", F1.mean().item())
