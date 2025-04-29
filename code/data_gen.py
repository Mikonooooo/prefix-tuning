from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy
import random


def gen_rand_subset(filepath, num_samples, outfilepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    subset = random.choices(lines, k=num_samples)
    with open(outfilepath, 'w') as f:
        f.writelines(subset)


if __name__ == "__main__":
    base = "data/e2e_data/"

    # Generate small train dataset
    filepath = base + "src1_train.txt"
    outfilepath = base + "small_train.txt"
    gen_rand_subset(filepath, 100, outfilepath)

    # Generate medium train dataset
    filepath = base + "src1_train.txt"
    outfilepath = base + "medium_train.txt"
    gen_rand_subset(filepath, 500, outfilepath)

    # Generate medium valid dataset
    filepath = base + "src1_valid.txt"
    outfilepath = base + "medium_val.txt"
    gen_rand_subset(filepath, 500, outfilepath)

    # Generate medium test dataset
    filepath = base + "src1_test.txt"
    outfilepath = base + "medium_test.txt"
    gen_rand_subset(filepath, 500, outfilepath)
