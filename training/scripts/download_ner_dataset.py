from datasets import load_dataset

print("Downloading ContractNER dataset...")

dataset = load_dataset("agilelab-org/ContractNER_Dataset")

print("\n================================")
print("ContractNER downloaded!")
print("================================")

print("\nDataset:")
print(dataset)

for split in dataset:
    print(f"\n{split}: {len(dataset[split])} examples")

print("\nColumns:")
print(dataset["train"].column_names)

print("\nFirst example:")
print(dataset["train"][0])