from datasets import load_dataset, Dataset
import json
import re
from collections import defaultdict

class RecapData():
    def __init__(self, filename = None, split = "validation"):
        '''
        This method initializes the class RecapData(). 

        @params 
            filename: file to load the data from, creates the data if None, default: None
            split: split of the dataset 
        '''
        self.mapped_summs = defaultdict(list) # mapped chapter summaries

        if filename:
            # read data from existing file
            self.mapped_summs[split] = self.read_data(filename)
        else:
            # create new dataset
            self.create_data()

    def create_data(self):
        '''
        This method creates a new dataset.
        '''
        self.kmfoda_data = load_dataset("kmfoda/booksum")
        
        for split in self.kmfoda_data.keys():
            if split != "train":
                self.mapped_summs[split] = self.map_chapters(split)

                # save dataset
                self.mapped_summs[split].to_json(f"./data/summ_{split}.jsonl")

    def map_chapters(self, split):
        '''
        This method maps the last chapter summaries to the second-to-last chapter summaries.

        @param split: split of the dataset
        @return last_prev_chapters: mapped chapter summaries
        '''
        # add the integer chapter number to the instances
        new_column = [None] * len(self.kmfoda_data[split])
        self.kmfoda_data[split] = self.kmfoda_data[split].add_column("chap_num", new_column)
        # add the start chapter number for aggregated summaries
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
        last_prev_chapters = Dataset.from_generator(self.concatenate_instances)
        return last_prev_chapters
    

    def concatenate_instances(self):
        '''
        This method concatenates the information about the second-to-last chapters and the last chapters.

        @yields dictionary of the mapped instance
        '''
        for instance in self.last_chapters:
            # find second-to-last chapters
            prev_chap = self.data_w_chap_num.filter(lambda batch: [prev_bid == instance["bid"] for prev_bid in batch["bid"]], batched=True).filter(lambda batch: [prev_chap_num == instance["start_chap_num"]-1 for prev_chap_num in batch["chap_num"]], batched=True)
            yield {"recap_id": f"{instance['bid']}_{instance['source']}", "bid": instance["bid"], "previous_summary_id": prev_chap["summary_id"], "previous_summary": prev_chap["summary_text"], "previous_source": prev_chap["source"], "next_summary_id": instance["summary_id"],"next_summary": instance["summary_text"], "next_source": instance["source"]}


    def get_last_chapter_num(self, batch):
        '''
        This function finds the highest chapter number of the book.

        @param batch: batch of instances
        '''
        for bid in batch["bid"]:
            if bid not in self.last_chap_nums.keys():
                self.last_chap_nums[bid] = max(self.data_w_chap_num.filter(lambda batch: [other_bid == bid for other_bid in batch["bid"]], batched=True)["chap_num"])
    
    def extract_chap_num(self, batch):
        '''
        This method extract the integer chapter numbers.

        @param batch: batch of instances
        @returns batch: batch of instances with the start and end chapter numbers
        '''
        nums = []
        start_nums = []
        for pos, summ_id in enumerate(batch["summary_id"]):
            # extract integer in summary id
            chap_num = re.search(r"\d+$", summ_id) 
            roman = re.search(r"[ivxlcdm]+$", summ_id)
            
            # extract start chapter for aggregated summaries
            start_chap_num = None
            start_roman = None
            if batch["is_aggregate"][pos]:
                start_chap_num = re.search(r"(\d+)-", summ_id) 
                start_roman = re.search(r"([ivxlcdm]+)-", summ_id)

            if chap_num:
                nums.append(int(chap_num.group()))
            elif roman:
                # transform to integer
                nums.append(self.tranform_roman_to_int(roman.group()))
            else:
                # refine regex if chapter number does not appear at the end of the summary id
                middle_chap = None
                middle_roman = None
                group = 0
                if batch["is_aggregate"][pos]:
                    middle_chap = re.search(r"-(\d+)", summ_id) 
                    middle_roman = re.search(r"-([ivxlcdm]+)", summ_id)
                    group = 1
                else:
                    middle_chap = re.search(r"\d+", summ_id)
                    middle_roman = re.search(r" [ivxlcdm]+ ", summ_id)


                if middle_chap:           
                    nums.append(int(middle_chap.group(group)))  
                elif middle_roman:
                    nums.append(self.tranform_roman_to_int(middle_roman.group(group).strip()))
                else:
                    # summary ids without chapter numbers
                    nums.append(None) 
            
            if start_chap_num:
                start_nums.append(int(start_chap_num.group(1)))
            elif start_roman:
                start_nums.append(self.tranform_roman_to_int(start_roman.group(1)))
            else:
                # if non-aggregated start_num == chap_num
                start_nums.append(nums[pos])

        batch["chap_num"] = nums
        batch["start_chap_num"] = start_nums
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

    def read_data(self, filename):
        '''
        This method reads an existing dataset.

        @param filename: path to the existing data file
        @returns HuggingFace dataset
        '''
        with open(filename, "r", encoding = "utf-8") as f:
            data = [json.loads(line) for line in f]
        return Dataset.from_list(data)
    
    def create_gold_data(self, split, filename):
        '''
        This method creates the reference recaps from the last chapter summaries.

        @params
            split: split of the data for the gold references
            filename: output file
        '''
        gold_summs = {idx: [summ] for idx, summ in zip(self.mapped_summs[split]["recap_id"], self.mapped_summs[split]["next_summary"])}
        
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(gold_summs, f, indent=4)

    def create_base_data(self, split, filename):
        '''
        This method creates the baseline recaps from the second-to-last chapter summaries.

        @params
            split: split of the data for the baseline recaps
            filename: output file
        '''
        base_summs = {idx: summs for idx, summs in zip(self.mapped_summs[split]["recap_id"], self.mapped_summs[split]["previous_summary"])}
        
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(base_summs, f, indent=4)


if __name__ == "__main__":
    # example
    example_recaps = RecapData("./data/example.jsonl", split = "validation")
    
    example_recaps.create_gold_data("validation", "./recaps/example/example_gold.json")
    example_recaps.create_base_data("validation", "./recaps/example/example_base.json")
