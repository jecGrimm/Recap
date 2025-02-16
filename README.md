# TODOS
## Allgemein
small experiment -> data processing -> development of hypers -> test -> visualizations -> code cleaning -> writing

### Nötig
1. small experiment
2. data processing
3. development
4. evaluation
5. analysis
6. code cleaning
-> CaptionMetrics auf git
-> remove unneccessary imports
-> remove unneccessary functions
-> documentation methoden
-> documentation readme
-> alles frisch laufen lassen
7. writing

### Extras
- analysis or summary better
- evaluation
    - meteor zum laufen kriegen
    - statistische Signifikanz
    - faktuality metric: summqa?

# Notes
## Usage:
installation:
- git clone --recurse-submodules <insert-url>
or after normal clone: 
- git submodule update --init --recursive

environment: 
- recap: python 3.13, Problem: pytorch kann nicht installiert werden
- recap3.7: Problem: urllib3 kann von Huggingface nicht ausgeführt werden
- recap3.11: benötigte Versionen aus DL file  -> geht mit NER

## Problems:
- Extract and align sequential chapter numbers 
-> is a string with various format (e.g. "act 3, scene 4" vs. "chapter 1-2")
-> summarization of multiple chapters at once ("chapter 1-2")
- The models do not produce the same results -> each created dataset might be different
- summaries are quite long -> similarities get very small
- Extractive approach
-> not very good readable (Sprünge, fehlende Zusammenhänge)
-> only the information is kept (keine schönen Bindeglieder und Einordnungen)
- when only using the previous chapter, there is some information missing (e.g. first and last names of the characters)
- only whole sentences are extracted, but only one part of them might be interesting
- treating the next chapter summary as input is not really correct, we should take the whole chapter, but the input is too small -> split approach?
- using the next chapter summary as gold reference is not correct because it is no recap
- LLM für Zusammenfassung:
-> Ist die Information korrekt? SummaC ist zwar eine factuality metric, aber die braucht zu lang mit dem Input (nicht mal ein Satz braucht 1min)
- limited input length (similarity 128) -> ends are cut off
- BookSum problems: 
-> data contamenation (but Gemma is filtered for evaluation datasets)
-> monolingual: English (Gemma also is mostly trained on english data)
-> quality of the summaries
-> harmful content in old books
- NER: Model is trained on news domain (CoNLL-2003)

## Experiment Setup
### Models
- Extractive
-> NER
-> DistilBERT
- Abstractive
-> Gemma-2-2b-it

### Evaluation
- No Bertscore because it depends on BERT embeddings and measures the similarity -> Not feasible to compare with the sentence similarity DistilBERT model 