#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os

os.environ["HF_HOME"] = "./.hf"
os.environ["TRANSFORMERS_CACHE"] = "./.hf"
os.environ["TRANSFORMERS_HOME"] = "./.hf"
os.environ["HF_CACHE"] = "./.hf"


import sys
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from accelerate import init_empty_weights, infer_auto_device_map
import argparse as ap

import pandas as pd

parser = ap.ArgumentParser()
parser.add_argument("data_split", type=str, help="data split to generate responses for. e.g.: test, val, test_jan25")
parser.add_argument("model_name", type=str, help="model name index to generate responses for. e.g.: 0, 1 or 2")

args = parser.parse_args()

if args.data_split == 'test_jan25': 
    file_path = f"data/{args.data_split}/mushroom.de-tst.v1.jsonl"
else:
    file_path = f"data/v2_splits/{args.data_split}/mushroom.de-val.v2.jsonl" # TODO: change according to needs

records = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        record = json.loads(line.strip())  # Parse each line as JSON
        if 'model_input' in record and 'id' in record:  # Ensure keys exist
            records.append({'id': record['id'], 'question': record['model_input'], 'model_id': record['model_id']})  # Extract 'id' and 'model_input'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[3]:


#pd.read_csv(file_path, sep='\t')


# In[4]:

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



# In[5]:

if args.model_name == '0':
    # model_name = 'togethercomputer/Pythia-Chat-Base-7B' 
    model_name = 'malteos/bloom-6b4-clp-german-oasst-v0.1' 

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

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map='balanced', trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # In[ ]:


    import tqdm as tqdm
    from transformers.utils import logging
    import pathlib
    logging.set_verbosity_warning()


    for shorthand, config in tqdm.tqdm(configs):
        output_file_path = os.path.join(f'./data/alt_res_{args.data_split}_setv2/alt_res/de', f'german-{model_name.split("/")[1]}.{shorthand}.jsonl')
        annotation_file_path = os.path.join(f'german-{model_name.split("/")[1]}-annotation.{shorthand}.jsonl')
        
        os.makedirs(f'./data/alt_res_{args.data_split}_setv2/alt_res/de', exist_ok=True)
        if not pathlib.Path(annotation_file_path).is_file():
            with open(output_file_path, 'w', encoding='utf-8') as file:
                for record in tqdm.tqdm(records):
                    if record['model_id'] != model_name:  # Skip if model_id doesn't match
                        continue
                    
                    id = record['id']
                    message = record['question']
                    prompt = message
                    
                    # Tokenize input
                    inputs = tokenizer(prompt, return_tensors="pt")
                    
                    # Generate 5 responses
                    outputs = model.generate(
                        **inputs,  # Inputs are created without explicit device specification
                        max_new_tokens=512,  # Adjust as needed
                        num_return_sequences=5,  # Generate 5 responses
                        no_repeat_ngram_size=2,
                        return_dict_in_generate=True,
                        output_logits=True,
                        do_sample=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                        **config,  # Sampling parameters (e.g., top_k, top_p, temperature)
                    )
                    
                    # Process each response
                    for response_index in range(5):  # Iterate over the 5 generated responses
                        response_text = tokenizer.decode(
                            outputs.sequences[response_index], skip_special_tokens=True
                        )
                        response_text = response_text.replace(prompt, "")  # Remove the input prompt from the output
                        response_token_ids = outputs.sequences[response_index].to("cpu").tolist()[len(inputs.input_ids[0]):]
                        response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)

                        # Extract logits for generated tokens
                        #logits_tensor = [l[response_index].to('cpu').tolist() for l in outputs.logits] # Logits for this response
                        response_logits = [l[response_index, token_id].item() for l, token_id in zip(outputs.logits, response_token_ids)]

                        assert len(response_tokens) == len(response_logits), f"Length mismatch: {len(response_tokens)} != {len(response_logits)}"
                        
                        # Prepare output record
                        response_record = {
                            "id": id,
                            "model_id": model_name,
                            "lang": "DE",
                            "response_index": response_index,  # Index of the current response
                            "output_text": response_text,
                            "output_tokens": response_tokens,
                            "output_logits": response_logits,
                        }
                        
                        # Save each response to the file
                        json.dump(response_record, file, ensure_ascii=False)
                        file.write('\n')

            
            # columns_to_extract = ['URL-de', 'lang', 'question', 'model_id', 'output_text', 'output_tokens', 'title']
            
            # output_data = []
            
            # with open(anootation_file_path, 'w', encoding='utf-8') as file:
            #     for data in records:
            #         extracted_data = {key: data[key] for key in columns_to_extract if key in data}
            
            #         json.dump(extracted_data, file, ensure_ascii=False)
            #         file.write('\n')


    # In[5]:


    del model
    torch.cuda.empty_cache()
    os.system('tput bel')



# In[5]:

else:
    from transformers import AutoTokenizer, MistralForCausalLM, set_seed
    model_name = "occiglot/occiglot-7b-de-en-instruct"
    model = MistralForCausalLM.from_pretrained(
        model_name, device_map='balanced', trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


    # In[9]:


    import tqdm as tqdm
    from transformers.utils import logging
    import pathlib
    logging.set_verbosity_warning()
    #model = model.to(device)

    for shorthand, config in tqdm.tqdm(configs):
        output_file_path = os.path.join(f'./data/alt_res_{args.data_split}_setv2/alt_res/de', f'german-{model_name.split("/")[1]}.{shorthand}.jsonl')
        anootation_file_path = f'german-{model_name.split("/")[1]}-annotation.{shorthand}.jsonl'

        if not pathlib.Path(anootation_file_path).is_file():
            with open(output_file_path, 'w', encoding='utf-8') as file:
                for record in tqdm.tqdm(records):
                    if record['model_id'] != model_name:  # Skip if model_id doesn't match
                        continue

                    messages = [
                    {"role": "system", 'content': 'You are a helpful assistant. Please give short and concise answers.'},
                    {"role": "user", "content": record['question']},
                    ]
                    inputs = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=True, 
                        add_generation_prompt=True, 
                        return_dict=False, 
                        return_tensors='pt',
                    )
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=256,
                        num_return_sequences=5,
                        no_repeat_ngram_size=2,
                        return_dict_in_generate=True,
                        output_logits=True,
                        do_sample=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                        # eos_token_id=tokenizer.encode('\n'),
                        # pad_token_id=tokenizer.encode('\n')[0],
                        **config,
                    )

                    id = record['id']
                    message = record['question']
                    prompt = message
                    
                    for response_index in range(5):
                        response_text = tokenizer.decode(outputs.sequences[response_index][len(inputs[0]):], skip_special_tokens=True)
                        # response_text = response_text.replace(prompt, "") # response repeats the input in the begining
                        response_token_ids = outputs.sequences[response_index].to("cpu").tolist()[len(inputs[0]):]
                        # response_embeddings = outputs.sequences[0].to("cpu").tolist()[len(inputs.input_ids[0]):]
                        response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)
                        logits_tensor = outputs.logits[response_index]  # Logits for this response
                        response_logits = [l[response_index, token_id].item() for l, token_id in zip(outputs.logits, response_token_ids)]
                        assert len(response_tokens) == len(response_logits), f"Length mismatch: {len(response_tokens)} != {len(response_logits)}"

                        # print("\n\n")
                        # print(f"Q: {message}")
                        # print(f"A: {response_text}")
                
                        # print("input length", len(inputs.input_ids[0]))
                        # # print("sequence length", len(outputs.sequences[0]))
                        # print("response token length", len(response_tokens))
                        # print("response token ID length", len(response_token_ids))
                        # print("logits length", len(response_logits))
                        # # print("embedding length", len(response_embeddings))
                        # raise

                        response_record = {
                            "id": id,
                            "model_id": model_name,
                            "lang": "DE",
                            "response_index": response_index,  # Index of the current response
                            "output_text": response_text,
                            "output_tokens": response_tokens,
                            "output_logits": response_logits,
                        }

                        # record['output_embeddings'] = response_embeddings
                
                        json.dump(response_record, file, ensure_ascii=False)
                        file.write('\n')
            # columns_to_extract = ['URL-de', 'lang', 'question', 'model_id', 'output_text', 'output_tokens', 'title']
            
            # output_data = []
            
            # with open(anootation_file_path, 'w', encoding='utf-8') as file:
            #     for data in records:
            #         extracted_data = {key: data[key] for key in columns_to_extract if key in data}
            
            #         json.dump(extracted_data, file, ensure_ascii=False)
            #         file.write('\n')
    os.system('tput bel')
    print("Script Completed")
