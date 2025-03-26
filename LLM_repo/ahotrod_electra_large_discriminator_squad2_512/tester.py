from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="ahotrod/electra_large_discriminator_squad2_512")

context = "The sky is blue."
question = "What color is the sky?"

result = qa_pipeline(question=question, context=context)