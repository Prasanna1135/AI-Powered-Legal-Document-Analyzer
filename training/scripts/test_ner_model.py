import spacy

print("Loading Legal NER model...")

nlp = spacy.load("training/saved_models/legal_ner_model")

print("Model loaded successfully!\n")

text = """
ABC Technologies Pvt. Ltd. appoints John Smith as Senior Manager.
His annual salary will be $120,000 effective January 1, 2026.
The agreement may be terminated on December 31, 2027.
"""

doc = nlp(text)

print("========== LEGAL NER RESULTS ==========")

print("\nOriginal Text:")
print(text)

print("\nDetected Entities:")

if doc.ents:
    for entity in doc.ents:
        print(
            f"Text: {entity.text} | "
            f"Label: {entity.label_}"
        )
else:
    print("No entities detected.")

print("\n======================================")