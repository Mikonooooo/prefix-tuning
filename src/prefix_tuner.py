from typing import List

import torch
from torch import nn
import torch.nn.functional as F
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

    def forward(self, input_ids, past_key_values=None, attention_mask=None, labels=None, use_cache=False):  # greedy
        batch_size = input_ids.shape[0]

        if past_key_values:
            prefix_key_values = past_key_values
        else:
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
            labels=labels,
            use_cache=use_cache
        )

        return output

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=200, eos_token=-1):
        input_ids = generate(self, input_ids, attention_mask,
                             max_new_tokens, eos_token)
        return input_ids


@torch.no_grad()
def generate(model, input_ids, attention_mask=None, max_new_tokens=200, eos_token=None):
    device = input_ids.device

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    outputs = model.forward(input_ids=input_ids, use_cache=True)
    past_key_values = outputs.past_key_values

    for _ in range(max_new_tokens):
        last_token = input_ids[:, -1:].to(device)
        outputs = model.forward(
            last_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True
        )

        past_key_values = outputs.past_key_values
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        next_token = torch.argmax(
            logits[-1, :], dim=-1, keepdim=True)  # [batch_size, 1]

        if next_token == eos_token:
            break

        input_ids = torch.cat((input_ids, next_token), dim=1)
        new_mask = torch.tensor(
            [1], dtype=torch.long, device=attention_mask.device).unsqueeze(1)
        attention_mask = torch.cat((attention_mask, new_mask), dim=1)

    return input_ids


@torch.no_grad()
def beam_search_generate(model, input_ids, beam_width=5, max_new_tokens=200, eos_token_id=None):
    device = input_ids.device
    model.eval()

    beams = [(input_ids, 0.0)]  # (token_sequence, score)

    for _ in range(max_new_tokens):
        new_beams = []

        for seq, score in beams:
            if eos_token_id is not None and seq[0, -1].item() == eos_token_id:
                new_beams.append((seq, score))
                continue

            outputs = model(seq)
            logits = outputs.logits[:, -1, :]  # shape: [1, vocab_size]
            log_probs = F.log_softmax(logits, dim=-1)

            topk_log_probs, topk_indices = torch.topk(
                log_probs, beam_width, dim=-1)

            for i in range(beam_width):
                next_token = topk_indices[0, i].unsqueeze(
                    0).unsqueeze(0)  # shape: [1,1]
                next_score = score + topk_log_probs[0, i].item()

                new_seq = torch.cat([seq, next_token], dim=1)
                new_beams.append((new_seq, next_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[
            :beam_width]

        if eos_token_id is not None and all(seq[0, -1].item() == eos_token_id for seq, _ in beams):
            break

    best_seq = beams[0][0]
    return best_seq


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


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

    filepath = "data/e2e_data/small_train.txt"
    examples = get_dict_from_data(filepath, tokenizer)

    for i, (k, v) in enumerate(examples.items()):
        tokenized = tokenizer(k, truncation=True, return_tensors='pt')
        print(tokenized["input_ids"])
        output_ids = model.generate(
            tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            eos_token=tokenizer.eos_token_id
        )
        # print(tokenizer.convert_ids_to_tokens(output_ids[0]))
        print(output_ids)
        start_idx = (output_ids[0] ==
                     tokenizer.bos_token_id).nonzero().flatten()[0]
        print(start_idx)
        print(tokenizer.decode(
            output_ids[0][start_idx:], skip_special_tokens=True))
        break
