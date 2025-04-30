from prefix_tuner import PrefixTuning, generate
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from data_load import get_dict_from_data
from tqdm import tqdm


def write_model_and_target_files(input_path, model, tokenizer, generated_path, target_path):
    """
    Processes a data file with lines of the form:
    table || target_sentence
    Groups by table, generates a model sentence per table, and writes:
    - generated_path: one generated sentence per line
    - target_path: all target sentences per table, with blank line between tables
    """
    table_to_targets = get_dict_from_data(input_path, tokenizer)

    # Generate sentences and write files
    with open(generated_path, 'w', encoding='utf-8') as gen_f, open(target_path, 'w', encoding='utf-8') as tgt_f:
        for table, targets in tqdm(table_to_targets.items()):
            # Model generation (replace with your model's inference code)
            # Example: encode table and generate
            inputs = tokenizer(table, return_tensors='pt')
            generated_ids = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs["attention_mask"],
                eos_token=tokenizer.eos_token_id
            )
            start_idx = (
                generated_ids[0] == tokenizer.bos_token_id).nonzero().flatten()[0]
            generated_sentence = tokenizer.decode(
                generated_ids[0][start_idx:], skip_special_tokens=True)
            gen_f.write(generated_sentence.strip() + '\n')

            tgt_f.writelines(tgt.strip() + '\n' for tgt in targets)
            tgt_f.write('\n')  # blank line between groups


if __name__ == "__main__":
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    prefix_model = PrefixTuning(model, prefix_len=5)
    prefix_model.init_P_weights(
        "models/e2e_prefix_prime.pth",
        "models/e2e_prefix_mlp.pth"
    )
    
    
    gen_filepath = "src/model-output.txt"
    label_filepath = "src/target.txt"
    input_filepath = "data/e2e_data/src1_test.txt"
    write_model_and_target_files(
        input_filepath, prefix_model, tokenizer, gen_filepath, label_filepath)
