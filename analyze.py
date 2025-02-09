from nltk.tokenize import sent_tokenize
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def kept_positions(summs, recaps):
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
    counter = 0
    num_kept = defaultdict(int)
    num_orig = defaultdict(int)
    for inst in dataset:
        for pos, summ in enumerate(inst["previous summary"][:3]):
            source = inst["previous source"][pos]

            num_kept[source] += len(sent_tokenize(recaps[counter]))
            num_orig[source] += len(sent_tokenize(summ))

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