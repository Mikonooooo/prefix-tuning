import torch
import torch.optim as optim
from data_load import make_dataloaders
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from prefix_tuner import PrefixTuning
from tqdm import tqdm
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import os

DATA_PATH = "data/e2e_data/"


def tune(tuner, dataset, num_epochs, batch_size, prefix_len, lr):
    if dataset == 'small':
        files = {"train": DATA_PATH + "small_train.txt",
                 "val": DATA_PATH + "medium_val.txt"}
    elif dataset == 'medium':
        files = {"train": DATA_PATH + "medium_train.txt",
                 "val": DATA_PATH + "medium_val.txt"}
    elif dataset == 'full':
        files = {"train": DATA_PATH + "src1_train.txt",
                 "val": DATA_PATH + "src1_valid.txt"}

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if tuner == "prefix":
        gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
        model = PrefixTuning(gpt_model, prefix_len=prefix_len)
    elif tuner == "fine":
        model = GPT2LMHeadModel.from_pretrained("gpt2")

    dataloaders = make_dataloaders(files, tokenizer, batch_size=batch_size)
    print(dataloaders)
    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"] if "val" in dataloaders.keys(
    ) else None

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    train_losses, val_losses = train(
        model, optimizer, train_dataloader, val_dataloader, num_epochs)

    return model, train_losses, val_losses


def train(model, optimizer, train_dataloader, val_dataloader=None, epochs=10):
    device = model.device
    train_losses = []
    val_losses = []

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=100, num_training_steps=epochs*len(train_dataloader))

    model.train()
    for i in range(epochs):
        epoch_loss = 0
        for batch in tqdm(train_dataloader):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)

            optimizer.zero_grad()
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss/len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f"epoch {i+1} loss: {avg_train_loss}")

        if val_dataloader is not None and i % 2 == 1:
            avg_eval_loss = eval(model, val_dataloader)
            print(avg_eval_loss)
            val_losses.append(avg_eval_loss)

    return train_losses, val_losses


def eval(model, eval_dataloader):
    device = model.device
    model.eval()

    running_loss = 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss

            running_loss += loss.item()

    model.train()
    return running_loss / len(eval_dataloader)


if __name__ == "__main__":

    # parse yaml
    hyperparameter_file = Path(__file__).parent / \
        "configs" / "hyperparameters.yaml"
    with open(hyperparameter_file, "r") as f:
        args = yaml.safe_load(f)

    model, train_losses, val_losses = tune(**args)
    print(args)

    torch.save(model.state_dict(), "models/e2e_prefix_tuned.pth")

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
