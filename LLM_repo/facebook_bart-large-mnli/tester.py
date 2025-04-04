from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
text = "This is a test sentence."
candidate_labels = ["politics", "sports", "technology"]
result = classifier(text, candidate_labels)