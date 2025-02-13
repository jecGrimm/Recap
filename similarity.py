# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
import json
from data import RecapData

class SentenceSimilarity():
    def __init__(self):
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v1')
        self.model = AutoModel.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v1')

    #Mean Pooling - Take attention mask into account for correct averaging
    # TODO: von Huggingface übernommen!
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # TODO: von Huggingface übernommen!
    def create_embeddings(self, sentences):
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, max pooling.
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings
    
    def compute_similarity(self, first_embed, second_embed):
        return F.cosine_similarity(first_embed, second_embed, dim =-1)
    
    def create_sentence_recap(self, dataset, treshold):
        recaps = dict()
        for instance in dataset:
            sim_recaps = []
            next_summ = instance["next_summary"]
            next_embed = self.create_embeddings([next_summ])

            for prev_summ in instance["previous_summary"]:
                sim_recap = ""
                for prev_sent in sent_tokenize(prev_summ):
                    prev_embed = self.create_embeddings([prev_sent])

                    cos_sim = self.compute_similarity(next_embed, prev_embed)
                    if cos_sim >= treshold:
                        sim_recap += prev_sent
                        sim_recap += " "

                sim_recaps.append(sim_recap.strip())
            recaps[instance["bid"]] = sim_recaps
        return recaps

    def store_recaps(self, filename, recaps):
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(recaps, f, indent=4)

if __name__ == "__main__":

    # Sentences we want sentence embeddings for
    sentences = ['This is an example sentence', 'Each sentence is converted. Das ist ein Paragraph, der aus mehreren Texten besteht']

    sim = SentenceSimilarity()
    
    summs = RecapData("./data/small_validation.jsonl", split = "validation")
    dataset = summs.mapped_summs["validation"]

    recaps = sim.create_sentence_recap(dataset, 0.2)
    sim.store_recaps("./recaps/small_sim.json", recaps)

