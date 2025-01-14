import torch
from transformers import RobertaTokenizerFast, RobertaForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
from evaluate import load

id2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 
            5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
label2id = {value: key for key, value in id2label.items()}

datasets = load_dataset("conll2003")

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large", 
                                                trust_remote_code=True, 
                                                add_prefix_space=True)
model = RobertaForTokenClassification.from_pretrained("roberta-large", 
                                                      num_labels=len(id2label), 
                                                      id2label=id2label, 
                                                      label2id=label2id, 
                                                      trust_remote_code=True)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding="max_length", is_split_into_words=True)
    labels = []
    
    for i, label in enumerate(examples["ner_tags"]):
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

training_args = TrainingArguments(
    output_dir="./results",
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
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained("./results/last")
tokenizer.save_pretrained("./results/last")
