from datasets import load_dataset
from collections import Counter

print("Loading ContractNER dataset...")

dataset = load_dataset("agilelab-org/ContractNER_Dataset")

train_data = dataset["train"]

print("\n========== DATASET ==========")
print(dataset)

print("\n========== COLUMNS ==========")
print(train_data.column_names)

print("\n========== NUMBER OF EXAMPLES ==========")
print(len(train_data))

print("\n========== FIRST EXAMPLE ==========")
print(train_data[0])

print("\n========== NER LABELS ==========")

all_labels = []

for example in train_data:
    for annotation in example["ner"]:
        start_token = annotation[0]
        end_token = annotation[1]
        label = annotation[2]

        all_labels.append(label)

label_counts = Counter(all_labels)

print("\nUnique labels:")
for label, count in label_counts.items():
    print(label, ":", count)