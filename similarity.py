# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize

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
    
    def create_sentence_recap(summs, sim, treshold):
        for instance in summs:
            sim_recaps = []
            next_summ = instance["next summary"]
            next_embed = sim.create_embeddings([next_summ])

            for prev_summ in instance["previous summary"][:3]:
                sim_recap = ""
                for prev_sent in sent_tokenize(prev_summ):
                    prev_embed = sim.create_embeddings([prev_sent])

                    cos_sim = sim.compute_similarity(next_embed, prev_embed)
                    if cos_sim >= treshold:
                        sim_recap += prev_sent
                        sim_recap += ". "

                sim_recaps.append(sim_recap)
            instance["similarity recap"] = sim_recaps

if __name__ == "__main__":

    # Sentences we want sentence embeddings for
    sentences = ['This is an example sentence', 'Each sentence is converted. Das ist ein Paragraph, der aus mehreren Texten besteht']

    similarity = SentenceSimilarity()
    # Load model from HuggingFace Hub
    #tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v1')
    #model = AutoModel.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v1')

    one_sentence = "This is a test"
    two_sentence = "This is a second test"
    one = similarity.create_embeddings(one_sentence)
    two = similarity.create_embeddings(two_sentence)
    print(similarity.compute_similarity(one, two))

    # sentence_embeddings = similarity.create_embeddings(sentences)

    # print("Sentence embeddings:")
    # print(sentence_embeddings)
    # print("Size:", sentence_embeddings.size())
    # print(similarity.compute_similarity(sentence_embeddings))