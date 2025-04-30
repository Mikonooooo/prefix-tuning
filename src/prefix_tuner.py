from typing import List

import torch
from torch import nn
from transformers import PreTrainedModel, GPT2PreTrainedModel, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, GPT2Model
from data_load import get_dict_from_data


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

    def init_P_weights(self, prime_weights, mlp_weights):
        self.P_prime.load_state_dict(torch.load(
            prime_weights, map_location=torch.device(self.device)))
        self.P_mlp.load_state_dict(torch.load(
            mlp_weights, map_location=torch.device(self.device)))

    def get_prefix(self, batch_size):
        prefix_indices = torch.arange(
            self.prefix_len, device=self.device).expand(batch_size, -1)
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

    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None):
        input_tokens = self.input_tokens.unsqueeze(
            0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(
            temp_control)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self, input_ids, attention_mask=None, labels=None):  # greedy
        batch_size = input_ids.shape[0]

        prefix_key_values = self.get_prefix(batch_size)

        if attention_mask is not None:
            prefix_attention_mask = torch.ones(
                batch_size, self.prefix_len, device=self.device)
            attention_mask = torch.cat(
                [prefix_attention_mask, attention_mask], dim=1)

        output = self.gpt_model(
            input_ids=input_ids,
            past_key_values=prefix_key_values,
            attention_mask=attention_mask,
            labels=labels
        )

        return output

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=200, eos_token=0):
        batch_size = input_ids.size(0)
        device = input_ids.device

        seq_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            outputs = self.forward(seq_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]

            next_token = torch.argmax(
                logits[:, -1, :], dim=-1, keepdim=True)  # [batch_size, 1]

            # Update stopped flags
            for i, token in enumerate(next_token.squeeze(1)):
                if token.item() == eos_token:
                    stopped[i] = True

            # Replace next_token with pad token eos for stopped sequences
            next_token[stopped] = eos_token

            # Append to sequences
            seq_ids = torch.cat((seq_ids, next_token), dim=1)

            # Update attention mask: 1 for new tokens, 0 if padded
            new_mask = (~stopped).long().unsqueeze(1)
            attention_mask = torch.cat((attention_mask, new_mask), dim=1)

            # Stop early if all sequences are done
            if stopped.all():
                break

        return seq_ids

        # batch_size = input_ids.shape[0]
        # prefix_key_values = self.get_prefix(batch_size)

        # output = self.gpt_model.generate(
        #     input_ids=input_ids,
        #     past_key_values=prefix_key_values,
        #     # attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     top_k=0,
        #     top_p=0.8,
        #     temperature=1,
        #     do_sample=True,
        #     # do_sample=False,
        #     num_return_sequences=1
        # )

        # return output


if __name__ == "__main__":
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # if you actually want to use it'
    model = PrefixTuning(gpt_model, prefix_len=5)
    model.init_P_weights(
        "models/e2e_prefix_prime.pth",
        "models/e2e_prefix_mlp.pth"
    )
    # model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    filepath = "data/e2e_data/src1_test.txt"
    examples = get_dict_from_data(filepath, tokenizer)

    for i, (k, v) in enumerate(examples.items()):
        tokenized = tokenizer(k, truncation=True, return_tensors='pt')
        output_ids = model.generate(
            tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            eos_token=tokenizer.eos_token_id
        )
        print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
        if i >= 9:
            break
