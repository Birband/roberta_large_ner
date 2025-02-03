from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from seqeval.metrics import f1_score, recall_score, precision_score
from trainer import tokenize_and_align_labels, load_and_map_dataset_kpwr
import sys
from tqdm import tqdm

id2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG',
            5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
label2id = {v: k for k, v in id2label.items()}

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

def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <EN|PL|Conll-EN|KPWR-PL>\n")
        exit(1)

    setting = sys.argv[1]

    if setting not in ["EN", "PL", "KPWR-PL", "Conll-EN"]:
        print(f"Usage: {sys.argv[0]} <EN|PL|Conll-EN|KPWR-PL>\n")
    
    dataset = None
    if setting in ["EN", "PL"]:
        dataset = load_dataset("Babelscape/wikineural")
    elif setting == "Conll-EN":
        dataset = load_dataset("conll2003")
    else:
        dataset = load_and_map_dataset_kpwr()

    model_name = ""
    test_split = ""
    if setting == "PL":
        model_name = "Birband/roberta_ner_pl"
        test_split="test_pl"
    elif setting == "KPWR-PL":
        model_name = "Birband/roberta_ner_pl"
        test_split="test"
    elif setting == "Conll-EN":
        model_name = "Birband/roberta_ner_eng"
        test_split="test"
    elif setting == "EN":
        model_name = "Birband/roberta_ner_eng"
        test_split="test_en"
    else:
        exit(1)

    test = dataset[test_split]

    model = RobertaForTokenClassification.from_pretrained(model_name)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

    tokenized_test = test.map(lambda examples: tokenize_and_align_labels(examples, tokenizer), batched=True)

    test_dataloader = DataLoader(tokenized_test, batch_size=8, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    all_preds = []
    all_labels = []

    for batch in tqdm(test_dataloader, desc="Processing batches"):
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