from datasets import load_dataset, Dataset
import json
import re
from collections import defaultdict
from tqdm import tqdm

# TODO: handle multiple chap-summs in the last chapter -> is aggregate
# TODO: handle act 3, scene 4 -> only use act as chap_num
# TODO: handle chapter number exceptions
# TODO: remove data without summary_id
# TODO: Doku
# TODO: remove unneccessary imports
class RecapData():
    def __init__(self):
        '''
        This method initializes the class RecapData(). 
        '''
        self.kmfoda_data = load_dataset("kmfoda/booksum")
        #print("text: ", kmfoda_data["train"][0]["summary_text"])
        self.mapped_summs = defaultdict(list)
        for split in self.kmfoda_data.keys():
            if split != "train":
                self.mapped_summs[split] = self.map_chapters(split)

                self.mapped_summs[split].to_json(f"./data/summ_{split}.jsonl")

    def map_chapters(self, split):
        # add the integer chapter number to the instances
        new_column = [None] * len(self.kmfoda_data[split])
        self.kmfoda_data[split] = self.kmfoda_data[split].add_column("chap_num", new_column)
        self.kmfoda_data[split] = self.kmfoda_data[split].add_column("start_chap_num", new_column)
        self.data_w_chap_num = self.kmfoda_data[split].map(self.extract_chap_num, batched=True)
        # filter out instances with default chapter number
        self.data_w_chap_num = self.data_w_chap_num.filter(lambda batch:[chap_num != None for chap_num in batch["chap_num"]], batched=True)

        # get the last chapter number for each instance
        self.last_chap_nums = defaultdict(int)
        self.data_w_chap_num.map(self.get_last_chapter_num, batched=True)

        # get only the last chapter instances
        self.last_chapters = self.data_w_chap_num.filter(lambda batch: [chap_num == self.last_chap_nums[batch["bid"][pos]] for pos, chap_num in enumerate(batch["chap_num"])], batched=True)
        
        # map the last chapters to the second to last chapter
        # TODO: Klappt das so? Oder muss ich das anders speichern?
        last_prev_chapters = Dataset.from_generator(self.concatenate_instances)
        return last_prev_chapters
    

    def concatenate_instances(self):
        for instance in self.last_chapters:
            prev_chap = self.data_w_chap_num.filter(lambda batch: [prev_bid == instance["bid"] for prev_bid in batch["bid"]], batched=True).filter(lambda batch: [prev_chap_num == instance["start_chap_num"]-1 for prev_chap_num in batch["chap_num"]], batched=True)
            yield {"bid": instance["bid"], "previous_summary_id": prev_chap["summary_id"], "previous_summary": prev_chap["summary_text"], "previous_source": prev_chap["source"], "next_summary_id": instance["summary_id"],"next_summary": instance["summary_text"], "next_source": instance["source"]}


    def get_last_chapter_num(self, batch):
        for bid in batch["bid"]:
            if bid not in self.last_chap_nums.keys():
                self.last_chap_nums[bid] = max(self.data_w_chap_num.filter(lambda batch: [other_bid == bid for other_bid in batch["bid"]], batched=True)["chap_num"])
    
    def extract_chap_num(self, batch):
        nums = []
        start_nums = []
        for pos, summ_id in enumerate(batch["summary_id"]):
            chap_num = re.search(r"\d+$", summ_id) 
            roman = re.search(r"[ivxlcdm]+$", summ_id)
            
            if batch["is_aggregate"][pos]:
                start_chap_num = re.search(r"(\d+)-", summ_id) 
                start_roman = re.search(r"([ivxlcdm]+)-", summ_id)

            if chap_num:
                nums.append(int(chap_num.group()))
            elif roman:
                # transform to integer
                nums.append(self.tranform_roman_to_int(roman.group()))
            else:           
                nums.append(None)   
                print("no chapter number: ", summ_id)
            
            if start_chap_num:
                start_nums.append(int(start_chap_num.groups(1)))
            elif start_roman:
                start_nums.append(self.tranform_roman_to_int(start_roman.groups(1)))
            else:
                start_nums.append(nums[pos])

        batch["chap_num"] = nums
        batch["start_chap_num"] = start_nums
        # TODO: handle other exceptions
        return batch
    
    def tranform_roman_to_int(self, roman):
        '''
        This method transforms roman numbers into integer.

        @param roman: roman number
        @returns number: integer of the roman number
        '''
        roman_to_int = {'i': 1, 'v': 5, 'x': 10, 'l': 50, 'c': 100, 'd': 500, 'm': 1000}
        number = 0
        for pos in range(len(roman)-1, -1, -1):
            char = roman[pos]
            char_num = 0
            if char in roman_to_int.keys():
                char_num = roman_to_int[char]

                if pos != len(roman)-1 and roman_to_int[roman[pos+1]] > char_num:
                    number -= char_num
                else:
                    number += char_num
        return number 


if __name__ == "__main__":
    recaps = RecapData()

    ### train, validation, test
    ## vorletzte chapter summary -> same book id und source
    ## letzte chapter summary -> same book id und source
    print("Done")
