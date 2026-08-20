from datasets import load_dataset
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
import os


print("======================================")
print("       LEGAL NER MODEL TRAINING       ")
print("======================================")


# Load dataset
print("\nLoading ContractNER dataset...")

dataset = load_dataset("agilelab-org/ContractNER_Dataset")

data = dataset["train"]

print("Dataset loaded!")
print("Total examples:", len(data))


# --------------------------------------------------
# Create spaCy training examples
# --------------------------------------------------

def create_training_example(example, nlp):

    tokens = example["tokenized_text"]
    annotations = example["ner"]

    # Keep token boundaries
    spaces = [True] * len(tokens)

    if len(spaces) > 0:
        spaces[-1] = False

    doc = spacy.tokens.Doc(
        nlp.vocab,
        words=tokens,
        spaces=spaces
    )

    entities = []

    for annotation in annotations:

        start_token = annotation[0]
        end_token = annotation[1]
        label = annotation[2]

        span = doc[start_token:end_token + 1]

        if span.text.strip():

            entities.append(
                (span.start_char, span.end_char, label)
            )

    # Remove overlapping entities
    valid_entities = []
    occupied = set()

    for start_char, end_char, label in entities:

        token_positions = set(
            range(
                doc.char_span(
                    start_char,
                    end_char
                ).start,
                doc.char_span(
                    start_char,
                    end_char
                ).end
            )
        )

        if not token_positions.intersection(occupied):

            valid_entities.append(
                (start_char, end_char, label)
            )

            occupied.update(token_positions)

    return Example.from_dict(
        doc,
        {
            "entities": valid_entities
        }
    )


# --------------------------------------------------
# Create NLP pipeline
# --------------------------------------------------

print("\nCreating spaCy pipeline...")

nlp = spacy.blank("en")

ner = nlp.add_pipe("ner")


# --------------------------------------------------
# Add labels
# --------------------------------------------------

print("\nAdding legal entity labels...")

labels = set()

for example in data:

    for annotation in example["ner"]:

        labels.add(annotation[2])


for label in sorted(labels):

    ner.add_label(label)


print("Entity types:", len(labels))

for label in sorted(labels):

    print("-", label)


# --------------------------------------------------
# Split dataset
# --------------------------------------------------

print("\nSplitting dataset...")

split_dataset = data.train_test_split(
    test_size=0.2,
    seed=42
)

train_data = split_dataset["train"]
test_data = split_dataset["test"]

print("Training examples:", len(train_data))
print("Testing examples:", len(test_data))


# --------------------------------------------------
# Convert training data
# --------------------------------------------------

print("\nPreparing training examples...")

train_examples = []

for i, example in enumerate(train_data):

    try:

        converted = create_training_example(
            example,
            nlp
        )

        train_examples.append(converted)

    except Exception as error:

        print(
            "Skipped example",
            i,
            ":",
            error
        )


print(
    "\nPrepared training examples:",
    len(train_examples)
)


# --------------------------------------------------
# CHECK THAT ENTITIES EXIST
# --------------------------------------------------

entity_count = 0

for example in train_examples:

    entity_count += len(example.reference.ents)


print(
    "Entities available for training:",
    entity_count
)


if entity_count == 0:

    print(
        "\nERROR: No entities were created."
    )

    raise SystemExit


# --------------------------------------------------
# Initialize model
# --------------------------------------------------

print("\nInitializing model...")

optimizer = nlp.initialize(
    get_examples=lambda: train_examples
)


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\nStarting NER training...")

random.seed(42)

epochs = 10

for epoch in range(epochs):

    random.shuffle(train_examples)

    losses = {}

    batches = minibatch(
        train_examples,
        size=compounding(
            4.0,
            16.0,
            1.001
        )
    )

    for batch in batches:

        nlp.update(
            batch,
            drop=0.2,
            sgd=optimizer,
            losses=losses
        )

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- Loss: {losses.get('ner', 0):.4f}"
    )


# --------------------------------------------------
# Save model
# --------------------------------------------------

model_path = "training/saved_models/legal_ner_model"

os.makedirs(
    "training/saved_models",
    exist_ok=True
)

nlp.to_disk(model_path)


print("\n======================================")
print("       NER TRAINING COMPLETE!")
print("======================================")

print("\nModel saved at:")

print(model_path)