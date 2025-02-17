#!/usr/bin/env python
# coding: utf-8

# Installing dependencies. You might need to tweak the CMAKE_ARGS for the `llama-cpp-python` pip package.

# In[1]:


# GPU llama-cpp-python; Starting from version llama-cpp-python==0.1.79, it supports GGUF
# !CMAKE_ARGS="-DLLAMA_CUBLAS=on " pip install 'llama-cpp-python>=0.1.79' --force-reinstall --upgrade --no-cache-dir
# For download the models
# !pip install huggingface_hub
# !pip install datasets


# We start by downloading an instruction-finetuned Mistral model.

# In[9]:

import argparse as ap
import os 

parser = ap.ArgumentParser()
parser.add_argument("data_split", type=str, help="data split to generate responses for. e.g.: test, val, test_jan25")

args = parser.parse_args()

print(args.data_split)
if args.data_split == 'test_jan25':
    file_path = f"data/{args.data_split}/mushroom.de-tst.v1.jsonl"
else:
    file_path = f"data/v2_splits/{args.data_split}/mushroom.de-val.v2.jsonl" # TODO: change according to needs

from huggingface_hub import hf_hub_download

model_name_or_path = "TheBloke/SauerkrautLM-7B-v1-GGUF"
model_basename = "sauerkrautlm-7b-v1.Q4_K_M.gguf"
#model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

# This config has been tested on an RTX 2080 (VRAM of 11GB).
# you might need to tweak with respect to your hardware.
"""from llama_cpp import Llama
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=4, # CPU cores
    n_batch=4096, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=250, # Change this value based on your model and your GPU VRAM pool.
    n_ctx=4096, # Context window
    logits_all=True
)"""


# In[7]:


import tqdm as tqdm
import json 
import csv



records = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        record = json.loads(line.strip())  # Parse each line as JSON
        if 'model_input' in record and 'id' in record:  # Ensure keys exist
            records.append({'id': record['id'], 'question': record['model_input'], 'model_id': record['model_id']})  # Extract 'id' and 'model_input'

from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, gguf_file=model_basename, device_map='auto', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, gguf_file=model_basename, trust_remote_code=True) 

import random

configs = [
    ('k50_p0.90_t0.1', dict(top_k=50, top_p=0.90, temperature=0.1)),
    ('k50_p0.95_t0.1', dict(top_k=50, top_p=0.95, temperature=0.1)),
    ('k50_p0.90_t0.2', dict(top_k=50, top_p=0.90, temperature=0.2)),
    ('k50_p0.95_t0.2', dict(top_k=50, top_p=0.95, temperature=0.2)),
    ('k50_p0.90_t0.3', dict(top_k=50, top_p=0.90, temperature=0.3)),
    ('k50_p0.95_t0.3', dict(top_k=50, top_p=0.95, temperature=0.3)),
    ('k75_p0.90_t0.1', dict(top_k=75, top_p=0.90, temperature=0.1)),
    ('k75_p0.95_t0.1', dict(top_k=75, top_p=0.95, temperature=0.1)),
    ('k75_p0.90_t0.2', dict(top_k=75, top_p=0.90, temperature=0.2)),
    ('k75_p0.95_t0.2', dict(top_k=75, top_p=0.95, temperature=0.2)),
    ('k75_p0.90_t0.3', dict(top_k=75, top_p=0.90, temperature=0.3)),
    ('k75_p0.95_t0.3', dict(top_k=75, top_p=0.95, temperature=0.3)),
]

random.shuffle(configs)

for shorthand, config in tqdm.tqdm(configs, desc='configs'):
    output_file_path = os.path.join(f'./data/alt_res_{args.data_split}_setv2/alt_res/de', f'german-{model_name_or_path.split("/")[1]}.{shorthand}.jsonl')
    os.makedirs(f'./data/alt_res_{args.data_split}_setv2/alt_res/de', exist_ok=True)

    with open(output_file_path, 'w') as file:
        for record in tqdm.tqdm(records, desc='items'):
            if record['model_id'] != model_name_or_path:
                continue
            
            id = record['id']
            message = record['question']
            prompt = f"[INST] {message} [/INST]"

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids#.to(model.device)

            """terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.encode('\n')[-1],
                ]"""


            outputs = model.generate(
                input_ids,
                max_new_tokens=512,
                num_return_sequences=5,
                #eos_token_id=terminators,
                pad_token_id=tokenizer.pad_token_id,
                attention_mask=inputs.attention_mask,
                return_dict_in_generate=True,
                output_logits=True,
                do_sample=True,
                **config
            )   

            for response_index in range(5):
                response = outputs.sequences[response_index]
                #print(outputs.sequences[response_index])
                response_text = tokenizer.decode(response, skip_special_tokens=True)
                response_text = response_text.split('[/INST]')[-1].strip()
                print(response_text)
                
                response_token_ids = response.to("cpu").tolist()
                response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)
                response_logits = [l[response_index, token_id].item() for l, token_id in zip(outputs.logits, response_token_ids)]

                #assert len(response_tokens) == len(response_logits), f"Length mismatch: {len(response_tokens)} != {len(response_logits)}"

                response_record = {
                        "id": id,   
                        "model_id": model_name_or_path,
                        "lang": "DE",
                        "response_index": response_index,  # Index of the current response
                        "output_text": response_text,
                        "output_tokens": response_tokens,
                        "output_logits": response_logits,
                }

                json.dump(response_record, file, ensure_ascii=False)
                file.write("\n")

os.system('tput bel')
print("Script Completed")
