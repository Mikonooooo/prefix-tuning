from pathlib import Path
import yaml

from transformers import PreTrainedModel, GPT2PreTrainedModel, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
import torch
from torch import nn

class PrefixTuning(GPT2PreTrainedModel):
    def __init__(self, config: Path, model: PreTrainedModel):
        super().__init__(model.config)

        # read config yaml
        with open(config, "r") as f:
            self.config = yaml.safe_load(f)
        self.prefix_length = self.config["prefix_length"]
        self.k = self.config["k"]
        self.hi = self.config["hi"]
        self.gpt2_config = GPT2Config()

        self.model = model
        self.p_prime = nn.Parameter(torch.randn(self.prefix_length, self.k))

        print(self.model.config)

        self.mlp = nn.Sequential(
            nn.Linear(self.k, self.k),
            nn.Tanh(),
            # nn.Linear(self.k, self.hi * 2 * )
        )

if __name__ == "__main__":
    # Example usage
    config_path = Path("code") / "configs" / "prefix_tuning.yaml"
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=GPT2Config())
    prefix_tuning = PrefixTuning(config_path, model)
