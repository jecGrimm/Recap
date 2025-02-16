from data import RecapData
from similarity import SentenceSimilarity
from ner import NER
from evaluate import evaluate
import json
from analyze import kept_positions, vis_num_kept, vis_pos, num_kept_sents
import os

# load data and models
test_recaps = RecapData("./data/summ_test.jsonl", split = "test")
dataset = test_recaps.mapped_summs["test"]

ner = NER()
sim = SentenceSimilarity()

# create recaps if neccessary
ner_file = "./recaps/test/test_ner.json"
if not os.path.isfile(ner_file):
    ner.treshold = 0 # best hyperparameter
    dataset.map(ner.create_ner_recap, batched = True)
    ner.store_recaps(ner_file, ner.recaps)

sim_file = "./recaps/test/test_sim.json"
if not os.path.isfile(sim_file):
    sim.treshold = 0.1 # best hyperparameter
    dataset.map(sim.create_sentence_recap, batched = True)
    sim.store_recaps(sim_file, sim.recaps)

gold_file = "./recaps/test/test_gold.json"
if not os.path.isfile(gold_file):
    test_recaps.create_gold_data("test", gold_file)

base_file = "./recaps/test/test_base.json"
if not os.path.isfile(base_file):
    test_recaps.create_base_data("test", base_file)

# Evaluation   
print("\nBaseline:\n")
base_metrics = evaluate(gold_file, base_file)

print("\nNER:\n")
ner_metrics = evaluate(gold_file, ner_file)

print("\nSentence Similarity:\n")
sim_metrics = evaluate(gold_file, sim_file)

print("\nLLM:\n")
llm_metrics = evaluate(gold_file, './recaps/test/test_llm.json')

# Store metrics in a file
with open("./evaluation/test_metrics.txt", 'w', encoding="utf-8") as eval_file:
    output = ""
    output += "\nBaseline:\n"
    for metric, score in base_metrics.items():
        output += f"{metric}: {score}\n"
    
    output += "\nNER:\n"
    for metric, score in ner_metrics.items():
        output += f"{metric}: {score}\n"

    output += "\nSentence Similarity:\n"
    for metric, score in sim_metrics.items():
        output += f"{metric}: {score}\n"

    output += "\nLLM:\n"
    for metric, score in llm_metrics.items():
        output += f"{metric}: {score}\n"
    
    eval_file.write(output)

# Analysis
# load recaps
with open(base_file, 'r') as file:
        base_res = json.load(file)
with open(ner_file, 'r') as file:
        ner_res = json.load(file)
with open(sim_file, 'r') as file:
        sim_res = json.load(file)

ner_recaps = [recap for recaps in ner_res.values() for recap in recaps]
sim_recaps = [recap for recaps in sim_res.values() for recap in recaps]
base_recaps = [summ for summs in base_res.values() for summ in summs]

# Positions of kept sentences
positions = {"NER": kept_positions(base_recaps, ner_recaps), "Similarity": kept_positions(base_recaps, sim_recaps)} 
vis_pos(positions, "./visualizations/test_pos.png")

# Proportion of kept sentences
src_names = list({idx.split("_")[1] for idx in base_res.keys()})
kept_sources = {"NER": num_kept_sents(dataset, ner_res, src_names), "Similarity": num_kept_sents(dataset, sim_res, src_names)}
vis_num_kept(kept_sources, src_names, "./visualizations/test_kept.png")