from datasets import load_dataset

print("Downloading ContractNLI dataset...")

dataset = load_dataset("presencesw/contract-nli")

print("\nDataset downloaded successfully!")
print(dataset)

for split in dataset:
    print(f"\n{split}: {len(dataset[split])} rows")

print("\nFirst example:")
print(dataset["train"][0])