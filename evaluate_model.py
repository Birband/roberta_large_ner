from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from seqeval.metrics import f1_score, recall_score, precision_score

# Mapowanie etykiet (zakładamy, że jest takie samo jak w CoNLL)
id2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG',
            5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
label2id = {v: k for k, v in id2label.items()}

# Funkcja do tokenizacji i wyrównywania etykiet
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=512,
        is_split_into_words=True
    )
    labels = []

    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # -100 jest ignorowane przez PyTorch
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Funkcja do obliczania metryk
def compute_metrics(preds, labels):
    preds = np.argmax(preds, axis=2)
    preds_list = []
    labels_list = []

    for pred, label in zip(preds, labels):
        preds_list.append([id2label[p] for p, l in zip(pred, label) if l != -100])
        labels_list.append([id2label[l] for l in label if l != -100])

    return {
        "f1": f1_score(labels_list, preds_list),
        "recall": recall_score(labels_list, preds_list),
        "precision": precision_score(labels_list, preds_list)
    }

# Funkcja do tworzenia batchy z paddowaniem
def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    # Padding do maksymalnej długości w batchu
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__ == '__main__':
    # Wczytanie datasetu Babelscape/wikineural
    dataset = load_dataset("Babelscape/wikineural")

    # Wybór odpowiedniego podziału (test_pl dla polskiego, test_eng dla angielskiego)
    model_name = "Birband/roberta_ner_eng"  # Zmień na "Birband/roberta_ner_eng" dla angielskiego
    if "pl" in model_name:
        test_split = "test_pl"
    else:
        test_split = "test_en"

    test = dataset[test_split]

    # Wczytanie modelu i tokenizera
    model = RobertaForTokenClassification.from_pretrained(model_name)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

    # Tokenizacja datasetu
    tokenized_test = test.map(tokenize_and_align_labels, batched=True)

    # Tworzenie DataLoader z funkcją collate_fn
    test_dataloader = DataLoader(tokenized_test, batch_size=8, collate_fn=collate_fn)

    # Przeniesienie modelu na GPU, jeśli dostępne
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Ewaluacja modelu
    model.eval()
    all_preds = []
    all_labels = []

    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        labels = batch["labels"]

        all_preds.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = compute_metrics(all_preds, all_labels)
    print("F1 score: ", metrics["f1"])
    print("Recall score: ", metrics["recall"])
    print("Precision score: ", metrics["precision"])