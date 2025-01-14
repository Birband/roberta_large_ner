from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from transformers import pipeline

# Load the local model and tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("results/last", add_prefix_space=True)
model = RobertaForTokenClassification.from_pretrained("results/last")

# Create the NER pipeline
nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Run the pipeline on a sample text
text = "Michael el Santane is my favourite New York artist"
ner_results = nlp_ner(text)

# Print the results
print(ner_results)
