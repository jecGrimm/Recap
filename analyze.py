from nltk.tokenize import sent_tokenize
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import json
from data import RecapData

def kept_positions(summs, recaps):
    '''
    This function counts the number of kept sentences in the beginning, middle and end of the summary.

    @params
        summs: original chapter summaries
        recaps: generated recaps
    @returns values: list with the positional counts of the kept sentences
    '''
    values = [0, 0, 0]
    for summ_pos, summ in enumerate(summs):
        sentences = sent_tokenize(summ)

        part = int(len(sentences)/3)

        if recaps[summ_pos] != "":
            for sent_pos, sent in enumerate(sentences, start=1):
                if sent in recaps[summ_pos]:
                    if sent_pos <= part:
                        values[0] += 1
                    elif sent_pos > part and sent_pos <= part*2:
                        values[1] += 1
                    else:
                        values[2] += 1
    return values

def num_kept_sents(dataset, recaps, src_names):
    '''
    This function counts the number of sentences kept per source.

    @params
        dataset: mapped chapter summaries
        recaps: generated recaps
        src_names: names of the summary sources
    @returns kept_normalized: list of the proportion of kept sentences per source
    '''
    num_kept = defaultdict(int)
    num_orig = defaultdict(int)
    for inst in dataset:
        for pos, summ in enumerate(inst["previous_summary"]):
            source = inst["previous_source"][pos]

            num_kept[source] += len(sent_tokenize(recaps[inst["recap_id"]][pos]))
            num_orig[source] += len(sent_tokenize(summ))

    kept_normalized = []
    for src in src_names:
        if num_orig[src] != 0:
            kept_normalized.append(num_kept[src]/num_orig[src])
        else:
            kept_normalized.append(0)

    return kept_normalized


def vis_pos(positions, filename):
    '''
    This function creates a plot of the number of sentences kept in the begin, middle and end of the summaries.

    @params
        positions: dictionary with the approach and the number of sentences per position
        filename: output file for the plot 
    '''
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
    ax.set_ylim(0, max([pos for pos_list in positions.values() for pos in pos_list])+50)

    #plt.show()
    plt.savefig(filename)

def vis_num_kept(kept_sources, names, filename):
    '''
    This function creates a plot for the proportion of sentences kept per source.

    @params
        kept_sources: dictionary with the number of kept sentences per source for each approach
        names: list of the source names
        filename: output file
    ''' 
    x = np.arange(len(names))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for approach, prop_kept in kept_sources.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, prop_kept, width, label=approach)
        multiplier += 1
            
    ax.set_ylabel('Proportion of the extracted sentences')
    ax.set_title('Extracted sentences per source')
    ax.set_xticks(x + width/2, names)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, top= 1)
    y_value=['{:.0%}'.format(x) for x in ax.get_yticks()]
    ax.set_yticklabels(y_value)

    #plt.show()
    plt.savefig(filename)

if __name__ == "__main__":
    # example
    # load data
    example_recaps = RecapData("./data/example.jsonl", split = "validation")
    dataset = example_recaps.mapped_summs["validation"]

    # load recaps
    with open('./recaps/example/example_base.json', 'r') as file:
        base_res = json.load(file)
    with open('./recaps/example/example_ner.json', 'r') as file:
            ner_res = json.load(file)
    with open('./recaps/example/example_sim.json', 'r') as file:
            sim_res = json.load(file)

    ner_recaps = [recap for recaps in ner_res.values() for recap in recaps]
    sim_recaps = [recap for recaps in sim_res.values() for recap in recaps]
    base_recaps = [summ for summs in base_res.values() for summ in summs]

    # Positions of kept sentences
    positions = {"NER": kept_positions(base_recaps, ner_recaps), "Similarity": kept_positions(base_recaps, sim_recaps)} 
    vis_pos(positions, "./visualizations/example_pos.png")

    # Proportion of kept sentences
    src_names = list({idx.split("_")[1] for idx in base_res.keys()})
    kept_sources = {"NER": num_kept_sents(dataset, ner_res, src_names), "Similarity": num_kept_sents(dataset, sim_res, src_names)}
    vis_num_kept(kept_sources, src_names, "./visualizations/example_kept.png")