import torch
from transformers import RobertaTokenizerFast, RobertaForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
from evaluate import load
import sys

from datasets import load_dataset

# Define id2label and label2id mappings
id2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 
            5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
label2id = {v: k for k, v in id2label.items()}

# Define category mapping
category_mapping = {
    "liv": "PER",
    "org": "ORG",
    "loc": "LOC",
    "fac": "MISC",
    "pro": "MISC",
    "oth": "MISC",
    "eve": "MISC",
    "adj": "MISC",
    "num": "MISC",
}


def load_and_map_dataset(dataset_name: str = "clarin-pl/kpwr-ner"):
    
    # Load the dataset
    dataset = load_dataset(dataset_name)
    
    def convert_label(label):
        """
        Convert a detailed dataset label to the simplified format.
        """
        # Get the original label string using id2label
        label_str = dataset["train"].features["ner"].feature.names[label]

        if label_str == 'O':
            return label2id['O']
        
        # Split into prefix and base entity type
        prefix, entity = label_str.split('-', 1)
        base_label = entity.split('_')[1]
        # Map to the simplified category using category_mapping
        mapped_category = category_mapping.get(base_label, 'O')  # Default to MISC
        simplified_label = f"{prefix}-{mapped_category}" if mapped_category != 'O' else f'{mapped_category}'
        return label2id[simplified_label]

    
    # Process the dataset
    def process_example(example):
        # Convert entity labels for a single example
        example['ner'] = [[convert_label(label) for label in labels] for labels in example['ner']]
        return example

    # Map processing over the dataset
    processed_dataset = dataset.map(process_example, batched=True)
    
    return processed_dataset

def TrainModel(path: str, language: str = "EN"):
    model = "roberta-large" if language == "EN" else "sdadas/polish-roberta-large-v2"

    id2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 
                5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
    label2id = {value: key for key, value in id2label.items()}

    datasets = load_dataset("conll2003") if language == "EN" else load_and_map_dataset()

    print("Data loaded successfully!\n")

    tokenizer = RobertaTokenizerFast.from_pretrained(model, 
                                                    trust_remote_code=True, 
                                                    add_prefix_space=True)
    model = RobertaForTokenClassification.from_pretrained(model, 
                                                        num_labels=len(id2label), 
                                                        id2label=id2label, 
                                                        label2id=label2id, 
                                                        trust_remote_code=True)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding="max_length", max_length=512, is_split_into_words=True)
        labels = []
        
        if "ner_tags" not in examples:
            ner_col_name = "ner"
        else:
            ner_col_name = "ner_tags"

        for i, label in enumerate(examples[ner_col_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

    print(tokenized_datasets['train']['labels'][:10])
    print("\n")
    print(tokenized_datasets['train']['input_ids'][:10])
    print("\n")
    print(tokenized_datasets['train']['tokens'][:10])


    training_args = TrainingArguments(
        output_dir=path,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation" if lang == "EN" else "test"],
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <EN|PL>\n")
        exit(1)

    lang = sys.argv[1]

    if lang not in ["EN", "PL"]:
        print(f"Usage: {sys.argv[0]} <EN|PL>\n")

    path = "results/ner_pl"

    TrainModel(path, lang)