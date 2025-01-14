from data import RecapData
from ner import NER
from similarity import SentenceSimilarity
from CaptionMetrics.pycocoevalcap.bleu.bleu import Bleu
from CaptionMetrics.pycocoevalcap.cider.cider import Cider
from CaptionMetrics.pycocoevalcap.meteor.meteor import Meteor
from CaptionMetrics.pycocoevalcap.rouge.rouge import Rouge
from CaptionMetrics.pycocoevalcap.spice.spice import Spice
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from summac.model_summac import  SummaCConv
from nltk.tokenize import word_tokenize

# TODO: Daten speichern
# TODO: in Funktionen auslagern
# TODO: in den Klassen unterbringen
# TODO: create_ner_recaps über mapping ausführen
# TODO: get_words deletes spaces
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

def evaluate(gold_file, result_file):

    with open(gold_file, 'r') as file:
        gts = json.load(file)

    with open(result_file, 'r') as file:
        res = json.load(file)

    # bleu_scorer = Bleu(n=4)
    # # # scorer += (hypo[0], ref1)   # hypo[0] = 'word1 word2 word3 ...'
    # # #                                 # ref = ['word1 word2 word3 ...', 'word1 word2 word3 ...']
    # bleu_score, bleu_scores = bleu_scorer.compute_score(gts, res)

    #print('bleu = %s' % bleu_score)

    # cider_scorer = Cider()
    # # scorer += (hypo[0], ref1)
    # (cider_score, cider_scores) = cider_scorer.compute_score(gts, res)
    # print('cider = %s' % cider_score)

    # meteor_scorer = Meteor()
    # meteor_score, meteor_scores = meteor_scorer.compute_score(gts, res)
    # print('meteor = %s' % meteor_score)

    # rouge_scorer = Rouge()
    # rouge_score, rouge_scores = rouge_scorer.compute_score(gts, res)
    # print('rouge = %s' % rouge_score)

    #TODO: test if this works now, otherwise use python 3.5
    spice_scorer = Spice()
    # spice_score is the average spice score (mean of all scores)
    spice_score, spice_scores = spice_scorer.compute_score(gts, res)
    #print("Länge Recap:", len(word_tokenize(gts["1"][0])))
    print('spice = %s' % spice_score)

def summacoz(tokenizer, model):
    pipe = pipeline("text2text-generation", 
                    model=model, 
                    tokenizer=tokenizer)

    PROMPT = """Is the hypothesis true based on the premise? Give your explanation afterwards.

    Premise: 
    {article}

    Hypothesis:
    {summary}
    """

    article = "Goldfish are being caught weighing up to 2kg and koi carp up to 8kg and one metre in length."
    summary = "Goldfish are being caught weighing up to 8kg and one metre in length."

    print(pipe(PROMPT.format(article=article, summary=summary), 
            do_sample=False, 
            max_new_tokens=512))
    """[{'generated_text': '\
    No, the hypothesis is not true. \
    - The hypothesis states that goldfish are being caught weighing up to 8kg and one metre in length. \
    - However, the premise states that goldfish are being caught weighing up to 2kg and koi carp up to 8kg and one metre in length. \
    - The difference between the two is that the koi carp is weighing 8kg and the goldfish is weighing 2kg.'}]"""

def summacconf(model):
    document = """Scientists are studying Mars to learn about the Red Planet and find landing sites for future missions.
    One possible site, known as Arcadia Planitia, is covered instrange sinuous features.
    The shapes could be signs that the area is actually made of glaciers, which are large masses of slow-moving ice.
    Arcadia Planitia is in Mars' northern lowlands."""

    summary1 = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers. This makes Arcadia Planitia ideal for future missions."
    score_conv1 = model.score([document], [summary1])
    print("[Summary 1] SummacConv score: %.3f" % (score_conv1["scores"][0])) # [Summary 1] SummaCZS Score: 0.582; SummacConv score: 0.536

def kept_positions(summs, recaps):
    values = [0, 0, 0]
    for summ_pos, summ in enumerate(summs):
        sentences = summ.split(".")
        #print("summ:", summ)
        #print("len summ:", len(sentences))
        part = int(len(sentences)/3)
        #print("part:", part)

        #print("pos:", summ_pos)

        #print("recap:", recaps[summ_pos])

        if recaps[summ_pos] != "":
            for sent_pos, sent in enumerate(sentences, start=1):
                #print("sent:", sent)
                if sent in recaps[summ_pos]:
                    #print("sentence is in recap")
                    if sent_pos <= part:
                        values[0] += 1
                    elif sent_pos > part and sent_pos <= part*2:
                        values[1] += 1
                    else:
                        values[2] += 1
    return values

def num_kept_sents(dataset, recaps, src_names):
    counter = 0
    num_kept = defaultdict(int)
    num_orig = defaultdict(int)
    for inst in dataset:
        for pos, summ in enumerate(inst["previous summary"][:3]):
            source = inst["previous source"][pos]

            num_kept[source] += len(recaps[counter].split("."))
            num_orig[source] += len(summ.split("."))

            counter += 1

    kept_norm = []
    for src in src_names:
        if num_orig[src] != 0:
            kept_norm.append(num_kept[src]/num_orig[src])
        else:
            kept_norm.append(0)

    return kept_norm


def vis_pos(positions):
    names = ['begin', 'middle', 'end']
    
    x = np.arange(len(names))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for approach, num_pos in positions.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, num_pos, width, label=approach)
        ax.bar_label(rects, padding=3)
        multiplier += 1
            
    ax.set_ylabel('Number of sentences')
    ax.set_title('Position of the recap sentences in the original summary')
    ax.set_xticks(x + width/2, names)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 12)

    plt.show()

def vis_num_kept(kept_sources, names):    
    x = np.arange(len(names))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for approach, prop_kept in kept_sources.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, prop_kept, width, label=approach)
        #ax.bar_label(rects, padding=3)
        multiplier += 1
            
    ax.set_ylabel('Proportion of the extracted sentences')
    ax.set_title('Extracted sentences per source')
    ax.set_xticks(x + width/2, names)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, top= 1)
    y_value=['{:.0%}'.format(x) for x in ax.get_yticks()]
    ax.set_yticklabels(y_value)

    plt.show()



#test_recaps = RecapData("./data/small_validation.jsonl")
#dataset = test_recaps.mapped_summs["validation"]

#sim = SentenceSimilarity()
#create_sentence_recap(test_recaps, sim, 0.2)

# Evaluation 

#print("\nOriginal summary:\n")
#evaluate('./small_gld_one.json', './small_prev.json')

# print("\nNER:\n")
# evaluate('./small_gld_one.json', './small_ner.json')

# print("\nSentence Similarity:\n")
# evaluate('./small_gld_one.json', './small_sim.json')

# print("\nLLM:\n")
# evaluate('./small_gld_three.json', './small_llm.json')

# SummaCoz
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
# model = AutoModelForSeq2SeqLM.from_pretrained("nkwbtb/flan-t5-11b-SummaCoz",
#                                             torch_dtype="auto",
#                                             device_map="auto")
# summacoz(tokenizer, model)

# SummaCConv
model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cpu", start_file="default", agg="mean")

summacconf(model_conv)

# Visualizations

# with open('./small_ner.json', 'r') as file:
#         ner_res = json.load(file)
# with open('./small_sim.json', 'r') as file:
#         sim_res = json.load(file)

# ner_recaps = [recap[0] for recap in ner_res.values()]
# sim_recaps = [recap[0] for recap in sim_res.values()]
# #print("ner values: ", [recap[0] for recap in ner_res.values()])

# Positions of kept sentences
# summs = [summ for inst in dataset for summ in inst["previous summary"][:3]]
# #print("summs:", summs)
# #print("recaps:", recaps)
# #print("dataset:", dataset)
# #print("dataset prev", dataset["previous summary"])
# #positions = {"NER": kept_positions(summs, ner_recaps), "Similarity": kept_positions(summs, sim_recaps)} 
# #vis_pos(positions)

# Proportion of kept sentences
#src_names = ["shmoop","cliffnotes","sparknotes"]

#kept_sources = {"NER": num_kept_sents(dataset, ner_recaps, src_names), "Similarity": num_kept_sents(dataset, sim_recaps, src_names)}
# #print("kept_sources", kept_sources)

#vis_num_kept(kept_sources, src_names)
