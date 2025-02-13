from CaptionMetrics.pycocoevalcap.bleu.bleu import Bleu
from CaptionMetrics.pycocoevalcap.cider.cider import Cider
from CaptionMetrics.pycocoevalcap.meteor.meteor import Meteor
from CaptionMetrics.pycocoevalcap.rouge.rouge import Rouge
from CaptionMetrics.pycocoevalcap.spice.spice import Spice
import json
from nltk.translate.bleu_score import corpus_bleu
from collections import defaultdict

def evaluate(gold_file, result_file):
    """
    This function evaluates the recaps in the result_file against the reference.

    @params 
        gold_file: reference file
        result_file: recap file
    @returns dictionary with the bleu, rouge and spice scores
    """
    ref_dict, hypo_dict = create_eval_dicts(gold_file, result_file)
    # TODO: testen!
    bleu_score = corpus_bleu(list(ref_dict.values()), [hypo[0] for hypo in hypo_dict.values()])
    print("bleu = %s" % bleu_score)

    rouge_scorer = Rouge()
    #rouge_score, _ = rouge_scorer.compute_score(gts, res)
    rouge_score, _ = rouge_scorer.compute_score(hypo_dict, ref_dict)
    print('rouge = %s' % rouge_score)

    spice_scorer = Spice()
    # spice_score is the average spice score (mean of all scores)
    spice_score, _ = spice_scorer.compute_score(hypo_dict, ref_dict)
    print('spice = %s' % spice_score)

    return {"BLEU": bleu_score, "ROUGE": rouge_score, "SPICE": spice_score}

def create_eval_dicts(gold_file, result_file):
    with open(gold_file, 'r') as file:
        gold = json.load(file)

    with open(result_file, 'r') as file:
        res = json.load(file)

    ref_dict = {}
    hypo_dict = {}

    if "llm" in result_file:
        gts = defaultdict(list)
        for idx, gld in gold.items():
            gts[idx.split("_")[0]].append(gld[0])

    for idx, recaps in res.items():
        i = 0
        for recap in recaps:
            ref_dict[f"{idx}_{i}"] = gts[idx]
            hypo_dict[f"{idx}_{i}"] = [recap]
        i += 1
    
    return ref_dict, hypo_dict
        


if __name__ == "__main__":
    gold_file = "./recaps/validation/validation_gold.json"
    result_file = "./recaps/validation/validation_base.json"
    ner_metrics = evaluate(gold_file, result_file)

    # with open("./evaluation_results.txt", 'w', encoding="utf-8") as eval_file:
    #     eval_file.write("NER:\n")
    #     for metric, score in ner_metrics.items():
    #         eval_file.write(f"{metric}: {score}\n")



