from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

print("Loading ContractNLI dataset...")

# Load dataset
dataset = load_dataset("presencesw/contract-nli")

data = dataset["train"]

# Combine contract text and hypothesis
texts = [
    sentence1 + " [SEP] " + sentence2
    for sentence1, sentence2 in zip(
        data["sentence1"],
        data["sentence2"]
    )
]

labels = data["gold_label"]

print("Dataset loaded!")
print("Total examples:", len(texts))

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("\nTraining examples:", len(X_train))
print("Testing examples:", len(X_test))

# Convert text into numerical TF-IDF features
print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 3),
    stop_words=None,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF conversion completed!")

# Train Logistic Regression model
print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=2000,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")

# Make predictions
print("\nEvaluating model...")

predictions = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy: {accuracy:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Create saved_models folder
os.makedirs("training/saved_models", exist_ok=True)

# Save model
joblib.dump(
    model,
    "training/saved_models/nli_classifier.pkl"
)

# Save vectorizer
joblib.dump(
    vectorizer,
    "training/saved_models/tfidf_vectorizer.pkl"
)

print("\n===================================")
print("MODEL SAVED SUCCESSFULLY!")
print("Location:")
print("training/saved_models/nli_classifier.pkl")
print("training/saved_models/tfidf_vectorizer.pkl")
print("===================================")