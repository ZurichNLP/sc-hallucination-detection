#!/usr/bin/env python
# coding: utf-8
import os
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, login
import argparse as ap
import json

parser = ap.ArgumentParser()
parser.add_argument("data_split", type=str, help="data split to generate responses for. e.g.: test, val, test_jan25")

args = parser.parse_args()

if args.data_split == 'test_jan25':
    file_path = f"data/{args.data_split}/mushroom.cs-tst.v1.jsonl"
else:
    file_path = f"data/v2_splits/{args.data_split}/mushroom.cs-val.v2.jsonl" # TODO: change according to needs


# Installing dependencies.
random.seed(2202)

# GPU
# This config has been tested on an v100. 32GB
# For download the models

os.environ['HF_HOME'] = './.hf/'
#!pip install --upgrade pip
#!pip install huggingface_hub
#!export HF_HOME='./.hf'
"""
os.makedirs('outputs/4annot', exist_ok=True)
os.makedirs('outputs/with_logits', exist_ok=True)"""


# safely copy your hf_token to this working directoy to login fo HF
with open('./.hf/token', 'r') as file:
    hftoken = file.readlines()[0].strip()

login(token=hftoken, add_to_git_credential=True)
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', trust_remote_code=True)# .to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

import pandas as pd
"""file_path = "questions-cs.tsv"
records = pd.read_csv(file_path, sep='\t').to_dict(orient='records')

pd.read_csv(file_path, sep='\t')"""

records = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        record = json.loads(line.strip())  # Parse each line as JSON
        if 'model_input' in record and 'id' in record:  # Ensure keys exist
            records.append({'id': record['id'], 'question': record['model_input'], 'model_id': record['model_id']})  # Extract 'id' and 'model_input'

configs = [
    ('k50_p0.90_t0.1', dict(top_k=50, top_p=0.90, temperature=0.1)),
    ('k50_p0.90_t0.2', dict(top_k=50, top_p=0.90, temperature=0.2)),
    ('k50_p0.90_t0.3', dict(top_k=50, top_p=0.90, temperature=0.3)),
    ('k50_p0.95_t0.1', dict(top_k=50, top_p=0.95, temperature=0.1)),
    ('k50_p0.95_t0.2', dict(top_k=50, top_p=0.95, temperature=0.2)),
    ('k50_p0.95_t0.3', dict(top_k=50, top_p=0.95, temperature=0.3)),
    ('k75_p0.90_t0.1', dict(top_k=75, top_p=0.90, temperature=0.1)),
    ('k75_p0.90_t0.2', dict(top_k=75, top_p=0.90, temperature=0.2)),
    ('k75_p0.90_t0.3', dict(top_k=75, top_p=0.90, temperature=0.3)),
    ('k75_p0.95_t0.1', dict(top_k=75, top_p=0.95, temperature=0.1)),
    ('k75_p0.95_t0.2', dict(top_k=75, top_p=0.95, temperature=0.2)),
    ('k75_p0.95_t0.3', dict(top_k=75, top_p=0.95, temperature=0.3)),
    ('default', dict()),
]

random.shuffle(configs)


import tqdm
from transformers.utils import logging
import pathlib
import json
logging.set_verbosity_warning()

for shorthand, config in tqdm.tqdm(configs):    
    output_file_path = os.path.join(f'./data/alt_res_{args.data_split}_setv2/alt_res/cs', f'czech-{model_name.split("/")[1]}.{shorthand}.jsonl')
    os.makedirs(f'./data/alt_res_{args.data_split}_setv2/alt_res/cs', exist_ok=True)
    
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for record in tqdm.tqdm(records):
            if record['model_id'] != model_name:  # Skip if model_id doesn't match
                continue
                
            message = [
                        {"role": "user", "content": record['question']},
                    ]

            inputs = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt"
            )#.to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                tokenizer.encode('\n')[-1],
            ]

            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                num_return_sequences=5,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_logits=True,
                do_sample=True,
                **config
            )

            id = record['id']
            message = record['question']
            prompt = message

            for response_index in range(5):
                response_text = tokenizer.decode(
                    outputs.sequences[response_index], skip_special_tokens=True
                ).strip()
                response_text = response_text.strip().split('\n\n')[2]
                print(response_text)
                response_token_ids = (
                    outputs.sequences[response_index].to("cpu").tolist()[len(inputs[0]) :]
                )
                response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)

                # Extract only the logits corresponding to the output tokens
                response_logits = [l[response_index, token_id].item() for l, token_id in zip(outputs.logits, response_token_ids)]

                assert len(response_tokens) == len(response_logits), f"Length mismatch: {len(response_tokens)} != {len(response_logits)}"

                response_record = {
                    "id": id,
                    "model_id": model_name,
                    "lang": 'CS',
                    "response_index": response_index,
                    "output_text": response_text,
                    "output_tokens": response_tokens,
                    "output_logits": response_logits,
                }

                json.dump(response_record, file, ensure_ascii=False)
                file.write('\n')
