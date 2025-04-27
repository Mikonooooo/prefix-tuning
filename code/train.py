import torch
from torch.utils.data import DataLoader
from data_gen import make_dataloaders
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def train(model, optimizer, criterion, dataloader, epochs=10):
    for i in range(epochs):
        for batch in dataloader:
            
            outputs = model()

            optimizer.zero_grad()
            loss = outputs.loss
            loss.backward()
            optimizer.step()


    return avg_loss


if __name__ == "__main__":
    files = {"small": "data/e2e_data/small_data.txt"}
    model     = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataloaders = make_dataloaders(files, tokenizer, batch_size=1)

    small_dataloader = dataloaders["small"]
    for inputs in small_dataloader:
        print(inputs["input_ids"].shape)
        outputs = model(**inputs)
        print(outputs["logits"].shape)
        break
