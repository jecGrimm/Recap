
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import notebook_login
import torch
from datasets import load_dataset
import re
from data import RecapData
from collections import defaultdict
from tqdm import tqdm
import json

# Please provide your HuggingFace access token and go to https://huggingface.co/google/gemma-2-2b to get access to Gemma-2-2b-it.
notebook_login()

"""1. Load Data and Model"""

# Load model
torch.cuda.empty_cache()

it_model_name = "google/gemma-2-2b-it"
it_model = AutoModelForCausalLM.from_pretrained(
    it_model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
it_model_tokenizer = AutoTokenizer.from_pretrained(it_model_name, trust_remote_code=True)

# Load mapped last chapters in BOOKSUM
test_summs = RecapData("/content/Recap/data/summ_test.jsonl", split = "test")
dataset = test_summs.mapped_summs["test"]

# Load BOOKSUM
kmfoda_test = load_dataset("kmfoda/booksum", split = "test")

# filter out any books without a last chapter
kmfoda_recap_summs = kmfoda_test.filter(lambda inst: [bid == inst["bid"] for bid in set(dataset["bid"])])

# fetch book titles
recap_book_titles = kmfoda_recap_summs.map(lambda inst: {"book_title": re.match(r"(.*?)\.", inst["book_id"]).group(1)})

# map book ids to titles
book_titles = set(zip(recap_book_titles["bid"], recap_book_titles["book_title"]))

# filter out double book titles
# 24 books in book_titles -> 21 in book_titles_cleaned
book_titles_cleaned =  set()
included_bids = set()
for bid, title in book_titles:
  if bid not in included_bids:
    book_titles_cleaned.add((bid, title))
    included_bids.add(bid)

"""2. Create Recaps"""

def create_messages(title):
  '''
  This function creates the prompts for the recap generation.

  @param title: title of one book
  @returns messages: list with three prompts
  '''
  messages = [
      {"role": "user", "content": f"Please generate a recap for the last chapter of the book {title}."},
      {"role": "user", "content": f"What do I need to know before reading the last chapter of the book {title}?"},
      {"role": "user", "content": f"Please summarize the book {title} excluding the last chapter."}
  ]
  return messages

def store_recaps(filename, recaps):
  '''
  This function saves the generated recaps in a file.

  @params
    filename: path of the output file
    recaps: generated recaps
  '''
  with open(filename, 'w', encoding="utf-8") as f:
      json.dump(recaps, f, indent=4)

def create_llm_recaps(book_titles):
  '''
  This function generates recaps with Gemma-2-2b-it for several books.

  @param book_titles: set of tuples with the book id and its title
  @returns recaps: dictionary with the generated recaps mapped to its book id
  '''
  recaps = defaultdict(list)

  for bid, title in tqdm(book_titles, desc="Generating recaps"):
    for message in create_messages(title):
      input_ids = it_model_tokenizer.apply_chat_template([message], return_tensors="pt", return_dict=True).to("cuda")
      outputs = it_model.generate(**input_ids, max_new_tokens=506)
      recap = it_model_tokenizer.decode(outputs[0])
      recaps[bid].append(recap)

  return recaps

# create recaps
llm_recaps = create_llm_recaps(book_titles_cleaned)
store_recaps("./test_llm.json", llm_recaps)