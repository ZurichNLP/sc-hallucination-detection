import json
import csv
import torch
import pandas as pd
import tqdm
import gzip
import shutil
import os
import sys
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument("data_split", type=str, help="data split to generate responses for. e.g.: test, val, test_jan25")
args = parser.parse_args()

set_seed(94326)

"""def read_data(file_path):
    with open(file_path, 'r') as istr:
        reader = csv.reader(istr)
        header = next(reader)
        records = [dict(zip(header, row)) for row in reader]

    return records"""

def load_model(model_name):
    with open('./.hf/token', 'r') as file:
        access_token = file.read()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", token = access_token
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,  token = access_token)
    return model, tokenizer

def main():
    """pathdata = "./fa-mushroom.test.csv"
    records = read_data(pathdata)"""
    if args.data_split == 'test_jan25':
        file_path = f"data/{args.data_split}/mushroom.fa-tst.v1.jsonl"
    else:
        file_path = f"data/v2_splits/{args.data_split}/mushroom.fa-{args.data_split}.v2.jsonl"

    model_name = "CohereForAI/aya-23-8B"

    records = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line.strip())  # Parse each line as JSON
            if 'model_input' in record and 'id' in record:  # Ensure keys exist
                records.append({'id': record['id'], 'question': record['model_input'], 'model_id': record['model_id']})  # Extract 'id' and 'model_input'   

    model, tokenizer = load_model(model_name)
    print(tokenizer.eos_token_id)
    configs = [
        ('k50_p0.90_t0.1', dict(top_k=50, temperature=0.1, top_p=0.90)),
        ('k50_p0.90_t0.2', dict(top_k=50, temperature=0.2, top_p=0.90)),
        ('k50_p0.90_t0.3', dict(top_k=50, temperature=0.3, top_p=0.90)),
        ('k50_p0.95_t0.1', dict(top_k=50, temperature=0.1, top_p=0.95)),
        ('k50_p0.95_t0.2', dict(top_k=50, temperature=0.2, top_p=0.95)),
        ('k50_p0.95_t0.3', dict(top_k=50, temperature=0.3, top_p=0.95)),
    ]
    """configs = [
        ('k50_t0.1', dict(top_k=50, temperature=0.1)),
        ('k50_t0.2', dict(top_k=50, temperature=0.2)),
        ('k50_t0.5', dict(top_k=50, temperature=0.5)),
        ('k50_t1.0', dict(top_k=50, temperature=1.0)),
        ('k75_t0.1', dict(top_k=75, temperature=0.1)),
        ('k75_t0.2', dict(top_k=75, temperature=0.2)),
        ('k75_t0.5', dict(top_k=75, temperature=0.5)),
        ('k75_t1.0', dict(top_k=75, temperature=1.0)),
        ('k100_t0.1', dict(top_k=100, temperature=0.1)),
        ('k100_t0.2', dict(top_k=100, temperature=0.2)),
        ('k100_t0.5', dict(top_k=100, temperature=0.5)),
        ('k100_t1.0', dict(top_k=100, temperature=1.0)),
    ]"""

    for shorthand, config in tqdm.tqdm(configs, desc='configs'):
        output_file_path = os.path.join(f'./data/alt_res_{args.data_split}_setv2/alt_res/fa', f'farsi-{model_name.split("/")[1]}.{shorthand}.jsonl')
        os.makedirs(f'./data/alt_res_{args.data_split}_setv2/alt_res/fa', exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as file:
            for record_ in tqdm.tqdm(records, desc='items'):
                record = {k: v for k, v in record_.items()}
                if record['model_id'] != model_name:
                    continue
                message = record['question'].strip()
                if not message:
                    continue
                message = [{"role": "user", "content": message}]

                inputs = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

                outputs = model.generate(
                    inputs,
                    max_new_tokens=512,
                    num_return_sequences=5,
                    return_dict_in_generate=True,
                    output_logits=True,
                    do_sample=True,
                    **config,
                )

                id = record['id']
                message = record['question']
                prompt = message

                for response_index in range(5):
                    response_text = tokenizer.decode(
                    outputs.sequences[response_index], skip_special_tokens=True
                    ).strip()
                    response_text = response_text.strip()
                    print(response_text)
                    response_token_ids = (
                        outputs.sequences[response_index].to("cpu").tolist()[len(inputs[0]) :]
                    )
                    response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)

                    # Extract only the logits corresponding to the output tokens
                    response_logits = [l[response_index, token_id].item() for l, token_id in zip(outputs.logits, response_token_ids)]

                    assert len(response_tokens) == len(response_logits), f"Length mismatch: {len(response_tokens)} != {len(response_logits)}"

                    response_record = {
                        'id': id,
                        'model_id': model_name,
                        'lang': 'FA',
                        'output_text': response_text,
                        'output_tokens': response_tokens,
                        'output_logits': response_logits
                    }

                    json.dump(response_record, file, ensure_ascii=False)
                    file.write('\n')
    del model
    torch.cuda.empty_cache()
    os.system('tput bel')
    print("Done")

if __name__ == "__main__":
    main()
