import joblib

print("Loading trained NLI model...")

model = joblib.load("training/saved_models/nli_classifier.pkl")
vectorizer = joblib.load("training/saved_models/tfidf_vectorizer.pkl")

print("Model loaded successfully!\n")


# ============================================================
# NLI TEST CASES
# ============================================================

test_cases = [

    {
        "name": "ENTAILMENT TEST",
        "contract": """
        The receiving party shall maintain all confidential information
        in strict confidence and shall not disclose such information
        to any third party.
        """,
        "hypothesis": "The receiving party must keep confidential information secret."
    },

    {
        "name": "CONTRADICTION TEST",
        "contract": """
        The receiving party shall not disclose confidential information
        to any third party.
        """,
        "hypothesis": "The receiving party is allowed to disclose confidential information to third parties."
    },

    {
        "name": "NOT MENTIONED TEST",
        "contract": """
        The receiving party shall maintain all confidential information
        in strict confidence.
        """,
        "hypothesis": "The receiving party must receive a salary of $100,000."
    }

]


# ============================================================
# MAKE PREDICTIONS
# ============================================================

for case in test_cases:

    contract_text = case["contract"].strip()
    hypothesis = case["hypothesis"]

    # Combine contract and hypothesis
    text = contract_text + " [SEP] " + hypothesis

    # Convert text using the same vectorizer used during training
    text_vectorized = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(text_vectorized)[0]

    print("=" * 70)
    print(case["name"])
    print("=" * 70)

    print("\nContract Text:")
    print(contract_text)

    print("\nHypothesis:")
    print(hypothesis)

    print("\nPredicted Label:", prediction)

    print()


print("=" * 70)
print("NLI TESTING COMPLETE")
print("=" * 70)