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
    # read in the references and hypotheses
    ref_dict, hypo_dict = create_eval_dicts(gold_file, result_file)

    bleu_score = corpus_bleu(list(ref_dict.values()), [hypo[0] for hypo in hypo_dict.values()])
    print("bleu = %s" % bleu_score)

    rouge_scorer = Rouge() # ROUGE-L
    rouge_score, _ = rouge_scorer.compute_score(ref_dict, hypo_dict)
    print('rouge = %s' % rouge_score)

    spice_scorer = Spice()
    # spice_score is the average spice score (mean of all scores)
    spice_score, _ = spice_scorer.compute_score(ref_dict, hypo_dict)
    print('spice = %s' % spice_score)

    return {"BLEU": bleu_score, "ROUGE": rouge_score, "SPICE": spice_score}

def create_eval_dicts(gold_file, result_file):
    '''
    This function creates the dictionaries needed for evaluation.

    @params
        gold_file: reference file
        result_file: hypotheses file
    @returns 
        ref_dict: dictionary with the references
        hypo_dict: dictionary with the hypotheses
    '''
    # load recaps 
    with open(gold_file, 'r') as file:
        gold = json.load(file)

    with open(result_file, 'r') as file:
        res = json.load(file)

    ref_dict = {}
    hypo_dict = {}
    gts = defaultdict(list)
    # extract book ids
    if "llm" in result_file:
        for idx, gld in gold.items():
            gts[idx.split("_")[0]].append(gld[0])
    else:
        gts = gold

    # transform recaps in dictionaries
    for idx, recaps in res.items():
        i = 0
        for recap in recaps:
            ref_dict[f"{idx}_{i}"] = gts[idx]
            hypo_dict[f"{idx}_{i}"] = [recap] # the evaluation metrics allow only one reference per time
            i += 1
    
    return ref_dict, hypo_dict
        


if __name__ == "__main__":
    # example
    gold_file = "./recaps/example/example_gold.json"
    base_file = "./recaps/example/example_base.json"
    base_metrics = evaluate(gold_file, base_file)

