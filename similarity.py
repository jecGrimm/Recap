# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
import json
from data import RecapData
from collections import defaultdict

class SentenceSimilarity():
    def __init__(self):
        '''
        This method initializes objects of the class SentenceSimilarity
        '''
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v1')
        self.model = AutoModel.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v1')

        self.recaps = defaultdict(list)
        self.treshold = 0.1 # best performance on validation data

    def mean_pooling(self, model_output, attention_mask):
        '''
        This method takes the attention mask into account for correct averaging. 
        Copied from https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v1.
        
        @params 
            model_output: output of the generated embeddings
            attention_mask: attention mask of the tokenized input
        @returns averaged sentence embeddings
        '''
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def create_embeddings(self, sentences):
        '''
        This method creates sentence embeddings.
        Copied from https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v1.
        
        @param sentences: list of sentences
        @returns sentence_embeddings: sentence representations
        '''
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, max pooling.
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings
    
    def compute_similarity(self, first_embed, second_embed):
        '''
        This method computes the cosine similarity of two embeddings.

        @params
            first_embed: sentence or paragraph embedding
            second_embed: sentence or paragraph embedding
        @returns cosine similarity between first_embed and second_embed
        '''
        return F.cosine_similarity(first_embed, second_embed, dim =-1)
    
    def create_sentence_recap(self, batch):
        '''
        This method creates recaps via sentence similarity.

        @param batch: batch of instances
        '''
        for pos, prev_summs in enumerate(batch["previous_summary"]):
            # last chpater embedding (whole paragraph)
            sim_recaps = []
            next_summ = batch["next_summary"][pos]
            next_embed = self.create_embeddings([next_summ])

            for prev_summ in prev_summs:
                sim_recap = ""
                for prev_sent in sent_tokenize(prev_summ):
                    # sentence embedding
                    prev_embed = self.create_embeddings([prev_sent])

                    cos_sim = self.compute_similarity(next_embed, prev_embed)
                    if cos_sim >= self.treshold:
                        sim_recap += prev_sent
                        sim_recap += " "

                sim_recaps.append(sim_recap.strip())
            self.recaps[batch["recap_id"][pos]] = sim_recaps

    def store_recaps(self, filename, recaps):
        '''
        This method stores the generated recaps in a file.

        @params
            filename: output file
            recaps: generated recaps
        '''
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(recaps, f, indent=4)

if __name__ == "__main__":
    # example
    sim = SentenceSimilarity()
    
    summs = RecapData("./data/example.jsonl", split = "validation")
    dataset = summs.mapped_summs["validation"]

    dataset.map(sim.create_sentence_recap, batched = True)

    sim.store_recaps("./recaps/example/example_sim.json", sim.recaps)

