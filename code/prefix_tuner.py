from typing import List

import torch
from torch import nn
from transformers import PreTrainedModel, GPT2PreTrainedModel, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, GPT2Model


class PrefixTuning(nn.Module):
    def __init__(self, model: GPT2Model, prefix_len: int = 10, k: int = 512):
        """
        GPT-2 that overrides embeddings with prefixes

        prefix_len: length of prefix
        k: dimension of the embedding for each entry in P' (small P)
        """
        super().__init__()
        self.gpt_model = model
        self.device = next(model.parameters()).device
        self.prefix_len = prefix_len
        self.hidden_dim = model.config.n_embd
        self.k = k
        self.n_layer = model.config.n_layer
        self.n_head = model.config.n_head
        self.head_dim = self.hidden_dim // self.n_head

        self.P_prime = nn.Embedding(
            self.prefix_len, self.hidden_dim).to(self.device)  # small P
        self.P_mlp = nn.Sequential(  # real P
            nn.Linear(self.hidden_dim, self.k),
            nn.Tanh(),
            nn.Linear(self.k, self.n_layer*2*self.hidden_dim)
        ).to(self.device)

        # freeze gpt2 parameters
        for param in self.gpt_model.parameters():
            param.requires_grad = False

    def get_prefix(self, batch_size):
        prefix_indices = torch.arange(self.prefix_len, device=self.device).expand(batch_size, -1)
        # batch_size x prefix_len x (n_layer * 2 * hidden_dim)
        prefix = self.P_mlp(self.P_prime(prefix_indices))

        # past_key_values should be length n_layer tuple of length 2 tuple of (batch_size, n_head, prefix_len, head_dim)
        prefix = prefix.view(batch_size, self.prefix_len,
                             2*self.n_layer, self.n_head, self.head_dim)
        # (2*n_layer, batch_size, num_heads, prefix_len, head_dim)
        prefix = prefix.permute((2, 0, 3, 1, 4))
        # tuple of len n_layer of tensors of shape (n_layer, batch_size, num_heads, prefix_len, head_dim)
        prefix = prefix.split(2)
        prefix = tuple((tensor[0], tensor[1]) for tensor in prefix)
        return prefix

    def forward(self, input_ids, attention_mask, labels):
        batch_size = input_ids.shape[0]

        prefix_key_values = self.get_prefix(batch_size)

        prefix_attention_mask = torch.ones(batch_size, self.prefix_len, device=self.device)
        attention_mask = torch.cat(
            [prefix_attention_mask, attention_mask], dim=1)

        output = self.gpt_model(
            input_ids=input_ids,
            past_key_values=prefix_key_values,
            attention_mask=attention_mask,
            labels=labels
        )
        return output


if __name__ == "__main__":
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prefix_model = PrefixTuning(model)  # if you actually want to use it'
    for k, v in prefix_model.named_parameters():
        print(k)

    # s = "Hamburger"
    # # 1) Tokenize the prompt
    # inputs = tokenizer(s, return_tensors="pt")
    # generated = inputs["input_ids"]  # shape [1, L]
    # past = None

    # # 2) Prime the model & get initial cache
    # outputs = model(input_ids=generated, use_cache=True)
    # past = outputs.past_key_values

    # # 3) Generate one token at a time
    # for _ in range(40):
    #     # Only pass in the very last token:
    #     last_token = generated[:, -1:].to(model.device)  # shape [1, 1]
    #     outputs = model(input_ids=last_token,
    #                     past_key_values=past,
    #                     use_cache=True)
    #     logits = outputs.logits[:, -1, :]
    #     past = outputs.past_key_values

    #     probs = torch.softmax(logits, dim=-1)
    #     next_token = torch.multinomial(probs, num_samples=1)

    #     generated = torch.cat((generated, next_token), dim=1)

    # print(tokenizer.decode(generated[0], skip_special_tokens=True))
