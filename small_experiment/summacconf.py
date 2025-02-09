# environment: summac3.11 with huggingface-hub 0.27.?
from summac.model_summac import SummaCConv, SummaCZS
import json
from tqdm import tqdm
from collections import defaultdict
from nltk.tokenize import sent_tokenize

# SummaCConv
#model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cpu") # If you have a GPU: switch to: device="cuda"

model = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cpu", start_file="default", agg="mean")

# document = """Scientists are studying Mars to learn about the Red Planet and find landing sites for future missions.
# One possible site, known as Arcadia Planitia, is covered instrange sinuous features.
# The shapes could be signs that the area is actually made of glaciers, which are large masses of slow-moving ice.
# Arcadia Planitia is in Mars' northern lowlands."""

# summary1 = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers. This makes Arcadia Planitia ideal for future missions."
#score_conv1 = 
#print("scores conv 1:", model.score([document], [summary1]))
#print("[Example] SummacConv score: " % (score_conv1["scores"][0])) # [Summary 1] SummaCZS Score: 0.582; SummacConv score: 0.536
#print("split test:", model.score(["test1", "der weiter geht"], ["noch ein test", "funzt das?"]))
# Small instances

with open("./small_gld_three.json", 'r') as file:
    gts = json.load(file)

with open("./small_llm.json", 'r') as file:
    res = json.load(file)

#print("gts:", gts)
#print("res:", res)

# 2551 max, 745 min
# 100: 10 sec
# 1000: 11 min
len_idx = 500
gts = {idx: [summ[0][:len_idx]] for idx, summ in gts.items()}
res = {idx: [summ[0][:len_idx]] for idx, summ in res.items()}

# split data
# num_splits = 10
# for idx, summ in gts.items():
#     text = summ[0]
#     #print("text", text)
#     split_len = int(len(text)/num_splits)
#     print("split len: ", split_len)
#     split_len_res = int(len(res[idx][0])/num_splits)

#     new_text = []
#     new_text_res = []
#     for i in range(1, num_splits+1):
#         new_text.append(text[split_len*(i-1):split_len*i])
#         new_text_res.append(res[idx][0][split_len_res*(i-1):split_len_res*i])
#     gts[idx] = new_text
#     res[idx] = new_text_res

# split sents
# gts = {idx: sent_tokenize(summ[0]) for idx, summ in gts.items()}
#res = {idx: sent_tokenize(summ[0]) for idx, summ in res.items()}

avg_score = 0.0
score_per_prompt = defaultdict(float)
for idx, recap in tqdm(gts.items(), desc="Computing SUMMACConf"):
    print("recap:", recap)
    #zs_score = model_zs.score(recap, res[idx])["scores"][0]
    #print("zs_scores: ", zs_score)
    #print(f"Score for BookID {idx}: {tmp_score}")
    for score in model.score(recap, res[idx])["scores"]:
        avg_score += score
        score_per_prompt[idx[-1]] += score

num_summs = len(gts.keys())
avg_score = avg_score/num_summs
avg_score_per_prompt = {idx: score/(num_summs/3) for idx, score in score_per_prompt.items()}

print("\nLLM:")
print("Average score over all summaries: ", avg_score)
print("Average scores for each prompt: ", avg_score_per_prompt)
