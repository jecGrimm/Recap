from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import json
from data import RecapData
from collections import defaultdict

class NER():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
        self.recaps = defaultdict(list)

    def get_words(self, ners):
        '''
        This method extracts the words from the recognized NERs.

        @param ners: list of NERs
        @returns words: set of words predicted as NERs
        '''
        words = set()

        current_word = ""
        for ner_obj in ners:
            if ner_obj["entity"][0] == "B":
                if current_word != "":
                    words.add(current_word.replace("#", ""))
                current_word = ner_obj["word"]
            else:
                current_word += ner_obj["word"]
        words.add(current_word.replace("#", ""))
        return words
    
    def create_ner_recap(self, batch, treshold = 0):
        '''
        @param dataset: split of mapped_summs data
        '''
        for pos, prev_summs in enumerate(batch["previous_summary"]):
            ner_recaps = []
            next_summ = batch["next_summary"][pos]
            next_ners = self.nlp(next_summ)
            next_words = self.get_words(next_ners)

            for prev_summ in prev_summs:
                ner_recap = ""
                for prev_sent in sent_tokenize(prev_summ):
                    prev_ners = self.nlp(prev_sent)

                    if prev_ners != []:
                        prev_words = self.get_words(prev_ners)

                        if len(next_words.intersection(prev_words)) > treshold:
                            ner_recap += prev_sent
                            ner_recap += " "
                ner_recaps.append(ner_recap.strip())
            
            self.recaps[batch["recap_id"][pos]] = ner_recaps

    def store_recaps(self, filename, recaps):
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(recaps, f, indent=4)

if __name__ == "__main__":
    ner = NER()
    # example = "My name is Wolfgang and I live in Berlin"

    # ner_results = ner.nlp(example)
    # print(ner_results)

    summs = RecapData("./data/small_validation.jsonl", split = "validation")
    dataset = summs.mapped_summs["validation"]

    dataset.map(ner.create_ner_recap, batched = True)
    #recaps = {idx: recaps for idx, recaps in zip(dataset["recap_id"], dataset["ner_recaps"])}
    ner.store_recaps("./recaps/small_ner.json", ner.recaps)
