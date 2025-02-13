from CaptionMetrics.pycocoevalcap.bleu.bleu import Bleu
from CaptionMetrics.pycocoevalcap.cider.cider import Cider
from CaptionMetrics.pycocoevalcap.meteor.meteor import Meteor
from CaptionMetrics.pycocoevalcap.rouge.rouge import Rouge
from CaptionMetrics.pycocoevalcap.spice.spice import Spice
import json
from nltk.translate.bleu_score import corpus_bleu

def evaluate(gold_file, result_file):
    """
    This function evaluates the recaps in the result_file against the reference.

    @params 
        gold_file: reference file
        result_file: recap file
    @returns dictionary with the bleu, rouge and spice scores
    """
    with open(gold_file, 'r') as file:
        gts = json.load(file)

    with open(result_file, 'r') as file:
        res = json.load(file)

    references = []
    ref_dict = {}
    hypos = []
    hypo_dict = {}
    for recap_id, recaps in res.items():
        i = 0
        for recap in recaps:
            hypos.append(recap)
            references.append(gts[recap_id])

            ref_dict[f"{recap_id}_{i}"] = gts[recap_id]
            hypo_dict[f"{recap_id}_{i}"] = [recap]
            i += 1
    bleu_score = corpus_bleu(references, hypos)
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

if __name__ == "__main__":
    gold_file = "./recaps/validation/validation_gold.json"
    result_file = "./recaps/validation/validation_base.json"
    ner_metrics = evaluate(gold_file, result_file)

    # with open("./evaluation_results.txt", 'w', encoding="utf-8") as eval_file:
    #     eval_file.write("NER:\n")
    #     for metric, score in ner_metrics.items():
    #         eval_file.write(f"{metric}: {score}\n")



