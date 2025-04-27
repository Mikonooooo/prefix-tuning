from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sys, os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
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
    def __init__(self, filepath, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.examples = []
        self.max_length = max_length

        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                table, text = line.split("||")
                table = ' {} {}'.format(table, tokenizer.bos_token)
                text = ' {} {}'.format(text, tokenizer.eos_token)   
                self.examples.append((table, text))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        table, text = self.examples[index]
        tokenized_table = self.tokenizer(table, truncation=True)
        tokenized_text = self.tokenizer(text, truncation=True)

        return {
            "input_ids": tokenized_table["input_ids"],
            "attention_mask": tokenized_table["attention_mask"],
            "labels": tokenized_text["input_ids"]
        }
    
def collate_fn(batch, pad_token_id):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]
    
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded
    }



if __name__ == "__main__":
    filepath = "data/e2e_data/src1_test.txt"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # get_dict_from_data(filepath, tokenizer)
    dataset = e2eDataset(filepath, tokenizer)
    print(dataset.__getitem__(0))

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id))
    for b in dataloader:
        print(b)
        break

