import torch
import torch.optim as optim
from data_load import make_dataloaders, gen_rand_subset
from output_gen import write_output_file
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from prefix_tuner import PrefixTuning
from tqdm import tqdm
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from data_load import get_dict_from_data
from train import tune
import os
import subprocess

DATA_PATH = "data/e2e_data/"

SIZES = [50, 100, 200, 500 ]

TRIALS = 5

# call on
def gen_small_data():
    for i in range(TRIALS):
        for size in SIZES:
            filepath = DATA_PATH + "src1_train.txt"
            outfilepath = f"{DATA_PATH}train_{size}_{i+1}.txt"
            gen_rand_subset(filepath, 100, outfilepath)
    

def train_small_data(args):
    for i in range(TRIALS):
        for size in SIZES:
            args["data_filepath"] = f"{DATA_PATH}train_{size}_{i+1}.txt"
            model, _, _ = tune(**args)
            print(args)

            if args['save_model']:
                torch.save(model.P_prime.state_dict(), f"models/e2e_prefix_prime_{size}_{i+1}.pth")
                torch.save(model.P_mlp.state_dict(), f"models/e2e_prefix_mlp_{size}_{i+1}.pth")

def test_small_data():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    input_filepath = "data/e2e_data/src1_test.txt"
    for i in range(TRIALS):
        for size in SIZES:
            # Load prefix-tuned model
            model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
            prefix_model = PrefixTuning(model, prefix_len=5, k=800)
            prefix_model.init_P_weights(
                "models/e2e_prefix_prime.pth",
                "models/e2e_prefix_mlp.pth"
            )
            
            gen_filepath = f"src/prefix-output_{size}_{i+1}.txt"
            label_filepath = f"src/prefix-target_{size}_{i+1}.txt"
            write_output_file(input_filepath, prefix_model, tokenizer, gen_filepath, label_filepath)


def eval_small_data():
    for i in range(TRIALS):
        for size in SIZES:
            result = subprocess.run([
                "./e2e_metrics/measure", f"src/target.txt", f"src/prefix-output_{size}_{i+1}.txt"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            ) 
            last_8_lines = result.stdout.strip().split('\n')[-8:]
            with open(f'evals/prefix_{size}_{i+1}_metrics.txt', 'w') as f:
                f.writelines(f"{line}\n" for line in last_8_lines)

if __name__ == "__main__":
    # parse yaml
    hyperparameter_file = Path(__file__).parent / \
        "configs" / "hyperparameters.yaml"
    with open(hyperparameter_file, "r") as f:
        args = yaml.safe_load(f)

    model, train_losses, val_losses = tune(**args)
    print(args)

    if args['save_model']:
        torch.save(model.P_prime.state_dict(), "models/e2e_prefix_prime.pth")
        torch.save(model.P_mlp.state_dict(), "models/e2e_prefix_mlp.pth")

    # plt.plot(train_losses)
    # plt.xlabel("epochs")
    # plt.ylabel("average loss")
    # plt.title("training loss on medium data")
    # plt.savefig("medium_train_losses.png")

    # plt.plot(val_losses)
    # plt.xlabel("epochs")
    # plt.ylabel("average loss")
    # plt.title("validation loss on medium data")
    # plt.savefig("medium_val_losses.png")

    # print(model.config)
    # print(model.device)
    # for inputs in small_dataloader:
    #     print(inputs["input_ids"].shape)
    #     outputs = model(**inputs)
    #     print(outputs.loss)
    #     break
