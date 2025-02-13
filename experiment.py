from data import RecapData
from similarity import SentenceSimilarity
from ner import NER
from evaluate import evaluate
import json
from analyze import kept_positions, vis_num_kept, vis_pos, num_kept_sents
import os

test_recaps = RecapData("./data/small_validation.jsonl", split = "validation")
dataset = test_recaps.mapped_summs["validation"]

if not os.path.isfile("./recaps/small/small_ner.json"):
    ner = NER()
    recaps = ner.create_ner_recap(dataset)
    ner.store_recaps("./recaps/small/small_ner.json", recaps)

if not os.path.isfile("./recaps/small/small_sim.json"):
    sim = SentenceSimilarity()
    recaps = sim.create_sentence_recap(dataset, 0.2)
    ner.store_recaps("./recaps/small/small_sim.json", recaps)

#Evaluation 
gold_file = "./recaps/small/small_gold.json"
print("\nBaseline:\n")
base_metrics = evaluate(gold_file, './recaps/small/small_base.json')

print("\nNER:\n")
ner_metrics = evaluate(gold_file, './recaps/small/small_ner.json')

print("\nSentence Similarity:\n")
sim_metrics = evaluate(gold_file, './recaps/small/small_sim.json')

print("\nLLM:\n")
llm_metrics = evaluate(gold_file, './recaps/small/small_llm.json')

# Store metrics in a file
with open("./evaluation/small_metrics.txt", 'w', encoding="utf-8") as eval_file:
    eval_file.write("\nBaseline:\n")
    for metric, score in base_metrics.items():
        eval_file.write(f"{metric}: {score}\n")
    
    eval_file.write("\nNER:\n")
    for metric, score in ner_metrics.items():
        eval_file.write(f"{metric}: {score}\n")

    eval_file.write("\nSentence Similarity:\n")
    for metric, score in sim_metrics.items():
        eval_file.write(f"{metric}: {score}\n")

    eval_file.write("\nLLM:\n")
    for metric, score in llm_metrics.items():
        eval_file.write(f"{metric}: {score}\n")

# Analysis
with open('./recaps/small/small_ner.json', 'r') as file:
        ner_res = json.load(file)
with open('./recaps/small/small_sim.json', 'r') as file:
        sim_res = json.load(file)

ner_recaps = [recap for recaps in ner_res.values() for recap in recaps]
sim_recaps = [recap for recaps in sim_res.values() for recap in recaps]

# Positions of kept sentences
# TODO: auf alle anpassen
summs = [summ for inst in dataset for summ in inst["previous summary"][:3]]
positions = {"NER": kept_positions(summs, ner_recaps), "Similarity": kept_positions(summs, sim_recaps)} 
vis_pos(positions, "./visualizations/small/small_pos.png")

# Proportion of kept sentences
src_names = ["shmoop","cliffnotes","sparknotes"]
kept_sources = {"NER": num_kept_sents(dataset, ner_recaps, src_names), "Similarity": num_kept_sents(dataset, sim_recaps, src_names)}
vis_num_kept(kept_sources, src_names, "./visualizations/small/small_kept.png")