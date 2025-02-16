from data import RecapData
from similarity import SentenceSimilarity
from ner import NER
from evaluate import evaluate
import os
from tqdm import tqdm
from CaptionMetrics.pycocoevalcap.rouge.rouge import Rouge
from CaptionMetrics.pycocoevalcap.spice.spice import Spice

# load data and models
val_recaps = RecapData("./data/summ_validation.jsonl", split = "validation")
dataset = val_recaps.mapped_summs["validation"]
tresholds = [0, 1, 2]
ner_metrics = []
sim_metrics = []

ner = NER()
sim = SentenceSimilarity()
rouge_scorer = Rouge()
spice_scorer = Spice()

for treshold in tqdm(tresholds, desc = "Evaluating tresholds"):
    ner_treshold = treshold
    sim_treshold = round(((treshold+1) * 0.1), 2) # needs to be rounded because treshold for 3 would be 0.30000000000000004

    ner_path = f"./recaps/validation/validation_ner_{ner_treshold}.json"
    sim_path = f"./recaps/validation/validation_sim_{sim_treshold}.json"

    # only create recaps if they do not exist yet
    if not os.path.isfile(ner_path):
        ner.treshold = ner_treshold
        dataset.map(ner.create_ner_recap, batched = True)
        ner.store_recaps(ner_path, ner.recaps)
    

    if not os.path.isfile(sim_path):
        sim.treshold = sim_treshold
        dataset.map(sim.create_sentence_recap, batched = True)
        sim.store_recaps(sim_path, sim.recaps)

    #Evaluation 
    gold_file = "./recaps/validation/validation_gold.json"

    print("\nNER:\n")
    ner_metrics.append(evaluate(gold_file, ner_path, rouge_scorer = rouge_scorer, spice_scorer = spice_scorer))

    print("\nSentence Similarity:\n")
    sim_metrics.append(evaluate(gold_file, sim_path, rouge_scorer = rouge_scorer, spice_scorer = spice_scorer))

# Store metrics in a file
with open("./evaluation/validation_metrics.txt", 'w', encoding="utf-8") as eval_file:
    output = ""
    
    for treshold in tresholds:
        output += f"\nNER with treshold {treshold}:\n"
        for metric, score in ner_metrics[treshold].items():
            output += f"{metric}: {score}\n"

        output += f"\nSentence Similarity with treshold {round(((treshold+1) * 0.1), 2)}:\n"
        for metric, score in sim_metrics[treshold].items():
            output += f"{metric}: {score}\n"
    
    eval_file.write(output)