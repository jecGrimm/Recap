from data import RecapData
from ner import NER
from similarity import SentenceSimilarity

# TODO: store sentence similarity in a file
# TODO: Daten speichern
# TODO: in Funktionen auslagern
# TODO: in den Klassen unterbringen
# TODO: create_ner_recaps über mapping ausführen
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

def create_ner_recap(dataset):
    '''
    @param dataset: split of mapped_summs data
    '''
    ner_model = NER()
    ner_column = [""] * len(dataset)
    recaps = dataset.add_column("ner recap", ner_column)

    for instance in recaps:
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

def create_sentence_recap(dataset, sim, treshold):
    #sim_column = [""] * len(dataset.mapped_summs["validation"])
    #recaps = dataset.add_column("similarity recap", sim_column)

    for instance in dataset.mapped_summs["validation"]:
        #print("instance:", instance)
        sim_recaps = []
        next_summ = instance["next summary"]
        print("next summary:", next_summ)
        next_embed = sim.create_embeddings([next_summ])

        for prev_summ in instance["previous summary"][:3]:
            sim_recap = ""
            #print("prev summ:", prev_summ)
            for prev_sent in prev_summ.split("."):
                print("prev sent:", prev_sent)
                prev_embed = sim.create_embeddings([prev_sent])

                cos_sim = sim.compute_similarity(next_embed, prev_embed)
                #print("cos sim:", cos_sim) 
                if cos_sim >= treshold:
                    sim_recap += prev_sent
                    sim_recap += ". "

            sim_recaps.append(sim_recap)
            print("recap: ", sim_recap)
        #instance["similarity recap"] = sim_recaps
    #print("recap dataset:", recaps)


test_recaps = RecapData("./data/small_validation.jsonl")
#dataset = test_recaps.mapped_summs["validation"]

sim = SentenceSimilarity()
create_sentence_recap(test_recaps, sim, 0.2)
