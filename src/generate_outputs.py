from prefix_tuner import PrefixTuning
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from data_load import get_dict_from_data, write_model_and_target_files


if __name__ == "__main__":
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    prefix_model = PrefixTuning(model, prefix_len=5)
    prefix_model.init_P_weights(
        "models/e2e_prefix_prime.pth",
        "models/e2e_prefix_mlp.pth"
    )
    write_model_and_target_files("data/e2e_data/src1_test.txt", prefix_model, tokenizer, "data/e2e_data/generated.txt", "data/e2e_data/reference.txt")
