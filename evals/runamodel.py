"""Run a GPT inherited model"""

from src.prefix_tuner import PrefixTuning
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


if __name__ == "__main__":
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    prefix_tuning_model = PrefixTuning(gpt2_model)


    ### model being run
    model = prefix_tuning_model
    ###


    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    s = "name : The Vaults | Type : pub | price : more than £ 30 | customer rating : 5 out of 5 | near : Café Adriatic"
    # 1) Tokenize the prompt
    inputs = tokenizer(s, return_tensors="pt")
    generated = inputs["input_ids"]  # shape [1, L]
    past = None
    attention_mask = inputs["attention_mask"]

    # 2) Prime the model & get initial cache
    outputs = model(input_ids=generated)
                    # attention_mask=attention_mask)
    past = outputs.past_key_values

    # 3) Generate one token at a time
    for _ in range(40):
        # Only pass in the very last token:
        last_token = generated[:, -1:].to(model.device)  # shape [1, 1]
        outputs = model(input_ids=generated)
                        # attention_mask=attention_mask
                        # past_key_values=past,
                        # use_cache=True)
        logits = outputs.logits[:, -1, :]
        past = outputs.past_key_values

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat((generated, next_token), dim=1)

        # attention_mask = torch.cat(
        #     (attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)),
        #     dim=1
        # )


    print(tokenizer.decode(generated[0], skip_special_tokens=True))