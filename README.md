# ROBERTA LARGE NER

## O modelach

### Trenowanie modeli

**Model angielski (Birband/roberta_ner_eng):**  
Został wytrenowany na zbiorze danych conll2003.

**Model polski (Birband/roberta_ner_pl):**  
Został wytrenowany na zbiorze danych KPWR-NER.

## Użycie modeli

### Ładowanie modelu i tokenizera

```python
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
```

**Dla modelu angielskiego:**

```python
tokenizer = RobertaTokenizerFast.from_pretrained("Birband/roberta_ner_eng", add_prefix_space=True)
model = RobertaForTokenClassification.from_pretrained("Birband/roberta_ner_eng")
```

**Dla modelu polskiego:**

```python
tokenizer = RobertaTokenizerFast.from_pretrained("Birband/roberta_ner_pl", add_prefix_space=True)
model = RobertaForTokenClassification.from_pretrained("Birband/roberta_ner_pl")
```

### Tworzenie pipeline NER

```python
from transformers import pipeline

nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
```

### Uruchomienie pipeline na przykładowym tekście

```python
text = "Michael el Santane is my favourite New York artist."

ner_results = nlp_ner(text)
print(ner_results)
```

## Wyniki modeli

Poniżej znajdują się wyniki metryk dla modeli Birband/roberta_ner_pl (polski) i Birband/roberta_ner_eng (angielski) na datasetach test_pl i test_eng z Babelscape/wikineural.

| Model                             | F1 Score | Recall | Precision |
|-----------------------------------|----------|--------|-----------|
| Birband/roberta_ner_pl (polski)   | 0.805     | 0.872   | 0.748      |
| Birband/roberta_ner_eng (angielski)| 0.844     | 0.857   | 0.831      |

## Jak uruchomić aplikację

### Sklonuj repozytorium:

```bash
git clone https://github.com/Birband/roberta_large_ner
cd twoje-repozytorium
```

### Zainstaluj zależności:

Upewnij się, że masz zainstalowane wymagane biblioteki. Możesz to zrobić za pomocą pip:

```bash
pip install -r requirements.txt
```