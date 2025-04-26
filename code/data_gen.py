from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sys, os
import torch
from torch.utils.data import Dataset
import numpy

def get_dict_from_data(filepath:str, tokenizer: GPT2Tokenizer) -> list[str]:
    input_output_dict = {}
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            input_table, output_sent = line.split("||")
            input_table = ' {} {}'.format(input_table, tokenizer.bos_token)
            output_sent = ' {} {}'.format(output_sent, tokenizer.eos_token)
            
            input_output_dict.setdefault(input_table, [])
            input_output_dict[input_table].append(output_sent) # append all sentences that map from table data
    return input_output_dict

class e2eDataset(Dataset):
    def __init__(self, filepath, tokenizer):
        self.tokenizer = tokenizer
        self.examples = []
        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                input_table, output_sent = line.split("||")
                input_table = ' {} {}'.format(input_table, tokenizer.bos_token)
                output_sent = ' {} {}'.format(output_sent, tokenizer.eos_token)
                
                self.examples.append({"table": input_table, "text": output_sent})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        table, text = example["table"], example["text"]
        table_tokens = self.tokenizer(table, return_tensors="pt")
        text_tokens = self.tokenizer(text, return_tensors="pt")

        table_len = len(table_tokens.input_ids[0])

        input_ids = torch.cat([table_tokens.input_ids.squeeze(0), text_tokens.input_ids.squeeze(0)])

        labels = input_ids.clone()
        labels[:table_len] = -100

        src_attn = torch.zeros_like(input_ids)
        src_attn[:table_len] = 1
        
        tgt_attn = torch.zeros_like(input_ids)
        tgt_attn[table_len:] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "src_attn": src_attn,
            "tgt_attn": tgt_attn,
            "src": table
        }









if __name__ == "__main__":
    filepath = "data/e2e_data/src1_test.txt"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # get_dict_from_data(filepath, tokenizer)
    dataset = e2eDataset(filepath, tokenizer)
    
