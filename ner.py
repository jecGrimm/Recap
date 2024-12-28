from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

class NER():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

if __name__ == "__main__":
    ner = NER()
    example = "My name is Wolfgang and I live in Berlin"

    ner_results = ner.nlp(example)
    print(ner_results)