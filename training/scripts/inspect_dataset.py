from datasets import load_dataset
from collections import Counter

print("Loading ContractNLI dataset...")

dataset = load_dataset("presencesw/contract-nli")

train_data = dataset["train"]

print("\n========== DATASET INFORMATION ==========")

print("Number of training examples:", len(train_data))

print("\nColumns:")
print(train_data.column_names)

print("\n========== LABEL DISTRIBUTION ==========")

labels = train_data["gold_label"]
label_counts = Counter(labels)

for label, count in label_counts.items():
    print(f"{label}: {count}")

print("\n========== SAMPLE ==========")

sample = train_data[0]

print("\nContract Text:")
print(sample["sentence1"][:1000])

print("\nHypothesis:")
print(sample["sentence2"])

print("\nGold Label:")
print(sample["gold_label"])