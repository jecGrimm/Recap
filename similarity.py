# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

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


    def create_embeddings(self, sentences, tokenizer, model):
        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling. In this case, max pooling.
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings
    
    def compute_similarity(self):
        return F.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim =-1)

if __name__ == "__main__":

    # Sentences we want sentence embeddings for
    sentences = ['This is an example sentence', 'Each sentence is converted. Das ist ein Paragraph, der aus mehreren Texten besteht']

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v1')
    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v1')

    sentence_embeddings = model.create_embeddings(sentences, tokenizer, model)

    print("Sentence embeddings:")
    print(sentence_embeddings)
    print("Size:", sentence_embeddings.size())
    print(F.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim =-1))