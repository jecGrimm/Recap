from data import RecapData
from similarity import SentenceSimilarity
from ner import NER
from evaluate import evaluate
import json
from analyze import kept_positions, vis_num_kept, vis_pos, num_kept_sents
import os

test_recaps = RecapData("./data/small_validation.jsonl", split = "validation")
dataset = test_recaps.mapped_summs["validation"]

ner = NER()
sim = SentenceSimilarity()

if not os.path.isfile("./recaps/small/small_ner.json"):
    ner.treshold = 0 # best hyperparameter
    dataset.map(ner.create_ner_recap, batched = True)
    ner.store_recaps("./recaps/small/small_ner.json", ner.recaps)

if not os.path.isfile("./recaps/small/small_sim.json"):
    sim.treshold = 0.1 # best hyperparameter
    dataset.map(sim.create_sentence_recap, batched = True)
    sim.store_recaps("./recaps/small/small_sim.json", sim.recaps)

#Evaluation 
gold_file = "./recaps/small/small_gold.json"
print("\nBaseline:\n")
base_metrics = evaluate(gold_file, './recaps/small/small_base.json')

print("\nNER:\n")
ner_metrics = evaluate(gold_file, './recaps/small/small_ner.json')

print("\nSentence Similarity:\n")
sim_metrics = evaluate(gold_file, './recaps/small/small_sim.json')

print("\nLLM:\n")
llm_metrics = evaluate(gold_file, './recaps/small/small_llm_test.json')

# Store metrics in a file
with open("./evaluation/small_metrics.txt", 'w', encoding="utf-8") as eval_file:
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
with open('./recaps/small/small_base.json', 'r') as file:
        base_res = json.load(file)
with open('./recaps/small/small_ner.json', 'r') as file:
        ner_res = json.load(file)
with open('./recaps/small/small_sim.json', 'r') as file:
        sim_res = json.load(file)

ner_recaps = [recap for recaps in ner_res.values() for recap in recaps]
sim_recaps = [recap for recaps in sim_res.values() for recap in recaps]
base_recaps = [summ for summs in base_res.values() for summ in summs]
# Positions of kept sentences
positions = {"NER": kept_positions(base_recaps, ner_recaps), "Similarity": kept_positions(base_recaps, sim_recaps)} 
vis_pos(positions, "./visualizations/small/small_pos.png")

# Proportion of kept sentences
src_names = list({idx.split("_")[1] for idx in base_res.keys()})
kept_sources = {"NER": num_kept_sents(dataset, ner_res, src_names), "Similarity": num_kept_sents(dataset, sim_res, src_names)}
vis_num_kept(kept_sources, src_names, "./visualizations/small/small_kept.png")