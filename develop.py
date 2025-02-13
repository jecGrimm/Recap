from data import RecapData
from similarity import SentenceSimilarity
from ner import NER
from evaluate import evaluate
import os

# Threads( StanfordCoreNLP ) Traceback (most recent call last):
#   File "/home/jana/Uni/Master_CL/WS2425/Summarization/Recap/develop.py", line 35, in <module>
#     ner_metrics.append(evaluate(gold_file, ner_path))
#                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/jana/Uni/Master_CL/WS2425/Summarization/Recap/evaluate.py", line 47, in evaluate
#     spice_score, _ = spice_scorer.compute_score(hypo_dict, ref_dict)
#                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/jana/Uni/Master_CL/WS2425/Summarization/Recap/CaptionMetrics/pycocoevalcap/spice/spice.py", line 70, in compute_score
#     subprocess.check_call(spice_cmd, 
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/jana/miniconda3/envs/recap3.11/lib/python3.11/subprocess.py", line 413, in check_call
#     raise CalledProcessError(retcode, cmd)
# subprocess.CalledProcessError: Command '['java', '-jar', '-Xmx8G', 'spice-1.0.jar', '/home/jana/Uni/Master_CL/WS2425/Summarization/Recap/CaptionMetrics/pycocoevalcap/spice/tmp/tmpi4axlui6', '-cache', '/home/jana/Uni/Master_CL/WS2425/Summarization/Recap/CaptionMetrics/pycocoevalcap/spice/cache', '-out', '/home/jana/Uni/Master_CL/WS2425/Summarization/Recap/CaptionMetrics/pycocoevalcap/spice/tmp/tmpjllybudv', '-subset', '-silent']' died with <Signals.SIGKILL: 9>.

val_recaps = RecapData("./data/summ_validation.jsonl", split = "validation")
dataset = val_recaps.mapped_summs["validation"]
tresholds = [0, 1, 2, 3, 4]
ner_metrics = []
sim_metrics = []

for treshold in tresholds:
    ner_treshold = treshold
    sim_treshold = treshold * 0.1

    ner_path = f"./recaps/validation/validation_ner_{ner_treshold}.json"
    sim_path = f"./recaps/validation/validation_sim_{sim_treshold}.json"

    if not os.path.isfile(ner_path):
        ner = NER()
        ner_recaps = ner.create_ner_recap(dataset)
        ner.store_recaps(ner_path, ner_recaps)
    

    if not os.path.isfile(sim_path):
        sim = SentenceSimilarity()
        sim_recaps = sim.create_sentence_recap(dataset, 0.2)
        ner.store_recaps(sim_path, sim_recaps)

    #Evaluation 
    gold_file = "./recaps/validation/validation_gold.json"

    print("\nNER:\n")
    ner_metrics.append(evaluate(gold_file, ner_path))

    print("\nSentence Similarity:\n")
    sim_metrics.append(evaluate(gold_file, sim_path))

# Store metrics in a file
with open("./evaluation/validation_metrics.txt", 'w', encoding="utf-8") as eval_file:
    output = ""
    
    for treshold in tresholds:
        output += f"\nNER with treshold {treshold}:\n"
        for metric, score in ner_metrics[treshold].items():
            output += f"{metric}: {score}\n"

        output += f"\nSentence Similarity with treshold {treshold * 0.1}:\n"
        for metric, score in sim_metrics[treshold].items():
            output += f"{metric}: {score}\n"