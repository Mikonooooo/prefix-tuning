from prefix_tuner import PrefixTuning, generate, beam_search_generate
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from data_load import get_dict_from_data
import torch
from tqdm import tqdm

device = "cuda" 

def write_target_file(input_path, target_path):
    print('creating target file')
    examples = {}
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            table, text = line.split("||")
            examples.setdefault(table, [])
            # append all sentences that map from table data
            examples[table].append(text)

    with open(target_path, 'w') as f:
        for texts in examples.values():
            f.writelines(text + '\n' for text in texts)
            f.write('\n')  # blank line between groups

    print('done')


def write_output_file(input_path, model, tokenizer, generated_path):
    """
    Processes a data file with lines of the form:
    table || target_sentence
    Groups by table, generates a model sentence per table, and writes:
    - generated_path: one generated sentence per line
    - target_path: all target sentences per table, with blank line between tables
    """
    table_to_targets = get_dict_from_data(input_path, tokenizer)

    # Generate sentences and write files
    with open(generated_path, 'w', encoding='utf-8') as gen_f:
        for table, targets in tqdm(table_to_targets.items()):
            # Model generation (replace with your model's inference code)
            # Example: encode table and generate
            inputs = tokenizer(table, return_tensors='pt')
            # generated_ids = generate(
            #     model.to(device),
            #     inputs['input_ids'].to(device),
            #     attention_mask=inputs["attention_mask"].to(device),
            #     max_new_tokens=200,
            #     eos_token=tokenizer.eos_token_id
            # )
            generated_ids = beam_search_generate(
                model,
                inputs['input_ids'].to(device),
                max_new_tokens=200,
                eos_token_id=tokenizer.eos_token_id
            )
            start_idx = (
                generated_ids[0] == tokenizer.bos_token_id).nonzero().flatten()[0]
            generated_sentence = tokenizer.decode(
                generated_ids[0][start_idx:], skip_special_tokens=True)
            # print(generated_sentence.strip())
            gen_f.write(generated_sentence.strip() + '\n')


            


if __name__ == "__main__":
    device = "cuda" 
    run_name = "paper-medium-ft_fine"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    input_filepath = "data/e2e_data/src1_test.txt"

    # Load prefix-tuned model
    # model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)
    # prefix_model = PrefixTuning(model, prefix_len=5, k=800)
    # prefix_model.init_P_weights(
    #     f"models/{run_name}_prime.pth",
    #     f"models/{run_name}_mlp.pth"
    # )
    # gen_filepath = f"evals/{run_name}.txt"
    # write_output_file(
    #     input_filepath, prefix_model, tokenizer, gen_filepath)

    ## Load finetuned model
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    model.load_state_dict(torch.load(
        f"models/{run_name}.pth", map_location="cuda"))  # or "cuda"
    gen_filepath = f"evals/{run_name}.txt"
    write_output_file(input_filepath, model, tokenizer, gen_filepath)
