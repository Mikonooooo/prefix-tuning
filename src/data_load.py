from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy
import random

def get_dict_from_data(filepath: str, tokenizer: GPT2Tokenizer) -> list[str]:
    examples = {}
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            table, text = line.split("||")
            table = ' {} {}'.format(table, tokenizer.bos_token)

            examples.setdefault(table, [])
            examples[table].append(text)
    return examples


class e2eDataset(Dataset):
    def __init__(self, filepath, tokenizer):
        self.tokenizer = tokenizer
        self.examples = []

        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                table, text = line.split("||")
                example = ' {} {} {} {}'.format(
                    table, tokenizer.bos_token, text, tokenizer.eos_token)
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        tokenized = self.tokenizer(self.examples[index], truncation=True)

        # mask out table in labels
        sep_idx = tokenized["input_ids"].index(self.tokenizer.bos_token_id)
        labels = tokenized["input_ids"].copy()
        labels[:sep_idx] = [-100]*sep_idx

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }


def collate_fn(batch, pad_token_id):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    input_ids_padded = pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask_padded = pad_sequence(
        attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded
    }


def make_dataloaders(files, tokenizer, batch_size):
    datasets = {}
    for name, filepath in files.items():
        if filepath is not None:
            dataset = e2eDataset(filepath, tokenizer)
            dataloader = DataLoader(dataset, batch_size, shuffle=True,
                                    collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id))
            datasets[name] = dataloader
    return datasets


    
    
    
    
