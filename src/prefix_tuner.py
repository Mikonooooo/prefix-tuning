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
        self.P_prime.load_state_dict(torch.load(prime_weights, map_location=torch.device(self.device)))
        self.P_mlp.load_state_dict(torch.load(mlp_weights, map_location=torch.device(self.device)))


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
    
    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None):
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size = input_ids.shape[0]

        prefix_key_values = self.get_prefix(batch_size)

        if attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, self.prefix_len, device=self.device)
            attention_mask = torch.cat(
                [prefix_attention_mask, attention_mask], dim=1)
            
            # print(attention_mask.shape)
            # print(input_ids.shape)

            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

            # print("attention mask", attention_mask[5])
            # print("decoded input_ids", tokenizer.decode(input_ids[5]))

        output = self.gpt_model(
            input_ids=input_ids,
            past_key_values=prefix_key_values,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = output.loss               # scalar float
        logits = output.logits           # shape [batch_size, seq_len, vocab_size]

        pred_ids = torch.argmax(logits, dim=-1)  # shape: [batch_size, seq_len]
        decoded_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_ids]

        return output, decoded_texts
    
    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=200):
        batch_size = input_ids.shape[0]
        prefix_key_values = self.get_prefix(batch_size)
        
        output = self.gpt_model.generate(
            input_ids=input_ids,
            past_key_values=prefix_key_values,
            # attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            top_k=0,
            top_p=0.8,
            temperature=1,
            do_sample=True,
            # do_sample=False,
            num_return_sequences=1
        )

        return output


if __name__ == "__main__":
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    prefix_model = PrefixTuning(model, prefix_len=5)  # if you actually want to use it'
    # prefix_model.load_state_dict(torch.load("models/e2e_prefix_tuned.pth", weights_only=True))
    prefix_model.init_P_weights(
        "models/e2e_prefix_prime.pth",
        "models/e2e_prefix_mlp.pth"
    )
    # prefix_model = GPT2LMHeadModel.from_pretrained("gpt2") # test GPT
    prefix_model.eval()

    filepath = "data/e2e_data/small_train.txt"
    examples = get_dict_from_data(filepath, tokenizer)
    for k, v in examples.items():
        print(k)
        print()
        tokenized = tokenizer(k, truncation=True, return_tensors='pt')
        # print(tokenizer.decode(tokenized["input_ids"][0]))
        print(tokenized["input_ids"][0])
        # sep_idx = tokenized["input_ids"].index(tokenizer.bos_token_id)
        # labels = tokenized["input_ids"].copy()
        # labels[:sep_idx] = [-100]*sep_idx
        
        # print("num of tokens", tokenized.input_ids.shape[1])
        # print("attention mask", tokenized.attention_mask)

        inputs = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            # "labels": labels
        }

        # use generate
        outputs = prefix_model.generate(**inputs)
        print("use generate")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

        # use forward
        for i in range(50):
            outputs, decoded_texts = prefix_model(**inputs)

            # Get logits and last predicted token
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # [batch_size, 1]

            # Append next token to input_ids
            inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)

            # Update attention mask
            new_mask = torch.ones_like(next_token)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], new_mask], dim=1)

            # Optionally print running generation
        print("use forward")
        print(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))

        break
    

    

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
