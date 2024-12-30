# TODOS
## Allgemein
- small experiment
- environment file
- data
- baseline
- sentence similarity
    - treshold entwickeln
- analysis or summary better?
- visualizations
- new dataset
- model
- evaluation
    - semantic based metric (spice zum laufen kriegen)
    - statistische Signifikanz

## Referat
- Grafiken
    - Wieviele Sätze werden behalten?
    - Evaluation
- Folien
- Paper zu Intention lesen

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