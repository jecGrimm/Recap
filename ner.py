from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from nltk.tokenize import sent_tokenize

class NER():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

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
    
    def create_ner_recap(self, dataset):
        '''
        @param dataset: split of mapped_summs data
        '''
        ner_column = [""] * len(dataset)
        recaps = dataset.add_column("ner recap", ner_column)

        for instance in recaps:
            ner_recaps = []
            next_summ = instance["next summary"]
            next_ners = self.nlp(next_summ)
            next_words = self.get_words(next_ners)

            #print("next ner:", next_ners)
            print("next words:", next_words)

            for prev_summ in instance["previous summary"][:3]:
                ner_recap = ""
                print("prev summ:", prev_summ)
                for prev_sent in sent_tokenize(prev_summ):
                    print("prev sent:", prev_sent)
                    prev_ners = self.nlp(prev_sent)

                    if prev_ners != []:
                        prev_words = self.get_words(prev_ners)

                        if len(next_words.intersection(prev_words)) != 0:
                            ner_recap += prev_sent
                            ner_recap += ". "
                ner_recaps.append(ner_recap)
            instance["ner recap"] = ner_recaps

if __name__ == "__main__":
    ner = NER()
    example = "My name is Wolfgang and I live in Berlin"

    ner_results = ner.nlp(example)
    print(ner_results)