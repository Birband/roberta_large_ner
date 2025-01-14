## ROBERTA LARGE NER

This model was trained on conll2003.

### Usage

#### Load model and tokenizer
```python
tokenizer = RobertaTokenizerFast.from_pretrained("results/last", add_prefix_space=True)
model = RobertaForTokenClassification.from_pretrained("results/last")
```

#### Create the NER pipeline
```python
nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
```

#### Run the pipeline on a sample text
```python
text = "Michael el Santane is my favourite New York artist."

ner_results = nlp_ner(text)
```