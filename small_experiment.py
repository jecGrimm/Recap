from data import RecapData
from ner import NER

# TODO: store sentence similarity in a file
# TODO: Daten speichern
# TODO: in Funktionen auslagern
# TODO: in den Klassen unterbringen
def get_words(ners):
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

test_recaps = RecapData("./data/small_validation.jsonl")
#print("data:", test_recaps.mapped_summs)
#print("prev summ:", len(test_recaps.mapped_summs["validation"][0]["previous summary"]))

#first_prev_summ = test_recaps.mapped_summs["validation"][0]["previous summary"][0]
#first_next_summ = test_recaps.mapped_summs["validation"][0]["next summary"]

#print("prev summ:", first_prev_summ)
#print("next summ:", first_next_summ)

def create_ner_recap(test_recaps):
    ner_model = NER()
    ner_column = [""] * len(test_recaps.mapped_summs["validation"])
    test_recaps.recaps["validation"] = test_recaps.mapped_summs["validation"].add_column("ner recap", ner_column)

    for instance in test_recaps.recaps["validation"]:
        ner_recaps = []
        next_summ = instance["next summary"]
        next_ners = ner_model.nlp(next_summ)
        next_words = get_words(next_ners)

        #print("next ner:", next_ners)
        print("next words:", next_words)

        for prev_summ in instance["previous summary"][:3]:
            ner_recap = ""
            print("prev summ:", prev_summ)
            for prev_sent in prev_summ.split("."):
                print("prev sent:", prev_sent)
                prev_ners = ner_model.nlp(prev_sent)

                if prev_ners != []:
                    prev_words = get_words(prev_ners)
                    #print("prev ner:", prev_ners)
                    #print("prev words:", prev_words)

                    if len(next_words.intersection(prev_words)) != 0:
                        ner_recap += prev_sent
                        ner_recap += ". "
            ner_recaps.append(ner_recap)
            print("recap:", ner_recap)
        instance["ner recap"] = ner_recaps
    
