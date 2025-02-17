import json
import csv
import sys
import torch
import random
import argparse as ap
import pandas as pd
import tqdm as tqdm
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_4bit=True)


parser = ap.ArgumentParser()
parser.add_argument("data_split", type=str, help="data split to generate responses for. e.g.: test, val, test_jan25")

args = parser.parse_args()

if args.data_split == 'test_jan25':
    file_path = f"data/{args.data_split}/mushroom.fi-tst.v1.jsonl"
else:
    file_path = f"data/v2_splits/{args.data_split}/mushroom.fi-val.v2.jsonl" # TODO: change according to needs

torch.cuda.empty_cache()

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        cache_dir="./.hf",
        quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./hf")
    print("Done loading!")
    return model, tokenizer


def write_records(record, file, logprobs_file, include_logits=False):
    records = record.copy()
    if include_logits:
        json.dump(records, logprobs_file, ensure_ascii=False)
        logprobs_file.write("\n")
    else:
        json.dump(records, file, ensure_ascii=False)
        file.write("\n")


def main():
    records = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line.strip())  # Parse each line as JSON
            if 'model_input' in record and 'id' in record:  # Ensure keys exist
                records.append({'id': record['id'], 'question': record['model_input'], 'model_id': record['model_id']})  # Extract 'id' and 'model_input'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model_name = "LumiOpen/Poro-34B-chat"
    model, tokenizer = load_model(model_name)

    save_records = []

    configs = [
        #('k50_p0.90_t0.1', dict(top_k=50, top_p=0.90, temperature=0.1)),
        #('k50_p0.95_t0.1', dict(top_k=50, top_p=0.95, temperature=0.1)),
        #('k50_p0.90_t0.2', dict(top_k=50, top_p=0.90, temperature=0.2)),
        #('k50_p0.95_t0.2', dict(top_k=50, top_p=0.95, temperature=0.2)),
        #('k50_p0.90_t0.3', dict(top_k=50, top_p=0.90, temperature=0.3)),
        ('k50_p0.95_t0.3', dict(top_k=50, top_p=0.95, temperature=0.3)),
        ('k75_p0.90_t0.1', dict(top_k=75, top_p=0.90, temperature=0.1)),
        ('k75_p0.95_t0.1', dict(top_k=75, top_p=0.95, temperature=0.1)),
        ('k75_p0.90_t0.2', dict(top_k=75, top_p=0.90, temperature=0.2)),
        ('k75_p0.95_t0.2', dict(top_k=75, top_p=0.95, temperature=0.2)),
        ('k75_p0.90_t0.3', dict(top_k=75, top_p=0.90, temperature=0.3)),
        ('k75_p0.95_t0.3', dict(top_k=75, top_p=0.95, temperature=0.3)),
    ]

    with torch.no_grad():
        for shorthand, config in tqdm.tqdm(configs, desc='configs'):
            output_file_path = os.path.join(f'./data/alt_res_{args.data_split}_setv2/alt_res/fi', f'finnish-{model_name.split("/")[1]}.{shorthand}.jsonl')
        
            os.makedirs(f'./data/alt_res_{args.data_split}_setv2/alt_res/fi', exist_ok=True)
            with open(output_file_path, 'w') as file:
                for record in tqdm.tqdm(records, desc='items'):
                    if record['model_id'] != model_name:
                        continue

                    #seed = random.randint(1, 10000)
                    set_seed(42)

                    #config_name, config = random.choice(configs)

                    message = record['question'].rstrip('\n')
                    message = [{"role": "user", "content": message}]
                    prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
                    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
                    output = model.generate(
                        input_ids,
                        max_new_tokens=256,
                        num_return_sequences=5,
                        return_dict_in_generate=True,
                        output_scores=True,
                        do_sample=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                        **config,
                        )
                    
                    id = record['id']
                    message = record['question']
                    prompt = message   

                    for response_index in range(5):
                        response_text = tokenizer.decode(output.sequences[response_index], skip_special_tokens=True)
                        response_text = response_text.replace(prompt, "") # response repeats the input in the begining

                        new_text_marker = "assistant\n"
                        marker_index = response_text.find(new_text_marker)
                        if marker_index != -1:
                            response_text = response_text[marker_index + len(new_text_marker):].strip()

                        response_token_ids = output.sequences[response_index].to("cpu").tolist()[len(input_ids[0]):]
                        response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)
                        
                        response_logits = []
                        for i, token_id in enumerate(response_token_ids):
                            token_logits = output.scores[i].to("cpu")[0, token_id].item()
                            response_logits.append(token_logits)

                        response_record = {
                        "id": id,
                        "model_id": model_name,
                        "lang": "FI",
                        "response_index": response_index,  # Index of the current response
                        "output_text": response_text,
                        "output_tokens": response_tokens,
                        "output_logits": response_logits,
                            }
                        json.dump(response_record, file, ensure_ascii=False)
                        file.write('\n')


if __name__ == "__main__":
    main()


torch.cuda.empty_cache()
os.system('tput bel')
