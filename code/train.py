import torch
from torch.utils.data import DataLoader
from data_gen import e2eDataset
from transformers import GPT2Tokenizer  
from transformers.data.data_collator import DataCollatorWithPadding


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
    filepath = "data/e2e_data/src1_test.txt"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    collator = DataCollatorWithPadding(tokenizer)
    dataset = e2eDataset(filepath, tokenizer)
    # print(dataset.__getitem__(0))
    dataloader = DataLoader(dataset, batch_size=3, collate_fn=collator)
    for batch in dataloader:
        print(batch)
        break