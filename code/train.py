import torch
import torch.optim as optim
from data_gen import make_dataloaders
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from prefix_tuner import PrefixTuning



def train(model, optimizer, dataloader, epochs=10):
    device = model.device
    model.train()

    for i in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)

            optimizer.zero_grad()
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss/len(dataloader)
        print(f"epoch {i+1} loss: {avg_loss}")

    return avg_loss


if __name__ == "__main__":
    files = {"small": "data/e2e_data/small_data.txt"}
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = PrefixTuning(gpt_model)

    dataloaders = make_dataloaders(files, tokenizer, batch_size=2)
    small_dataloader = dataloaders["small"]

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    num_epochs = 3

    train(model, optimizer, small_dataloader, num_epochs)

    # print(model.config)
    # print(model.device)
    # for inputs in small_dataloader:
    #     print(inputs["input_ids"].shape)
    #     outputs = model(**inputs)
    #     print(outputs.loss)
    #     break
