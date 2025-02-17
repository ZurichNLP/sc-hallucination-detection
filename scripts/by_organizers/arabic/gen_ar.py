import os
import sys
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm as tqdm
from transformers.utils import logging
import json
import argparse as ap

logging.set_verbosity_warning()

seed = 42
torch.manual_seed(seed)

os.environ["HF_HOME"] = "./.hf"
os.environ["TRANSFORMERS_CACHE"] = "./.hf"
os.environ["TRANSFORMERS_HOME"] = "./.hf"
os.environ["HF_CACHE"] = "./.hf"

os.environ['HF_HOME'] = './.hf/'

parser = ap.ArgumentParser()
parser.add_argument("data_split", type=str, help="data split to generate responses for. e.g.: test, val, test_jan25")
parser.add_argument("model_name", type=str, help="model name to generate responses for. e.g.: Iker/Llama-3-Instruct-Neurona-8b-v2, meta-llama/Meta-Llama-3-8B-Instruct, TheBloke/Mistral-7B-Instruct-v0.2-GGUF, occiglot/occiglot-7b-de-en-instruct, SeaLLMs/SeaLLM-7B-v2.5, openchat/openchat-3.5-0106-gemma, arcee-ai/Arcee-Spark")

args = parser.parse_args()

if args.data_split == 'val':
    file_path = f"data/v2_splits/{args.data_split}/mushroom.ar-val.v2.jsonl" # TODO: change according to needs
elif args.data_split == 'test_jan25':
    file_path = f"data/{args.data_split}/mushroom.ar-tst.v1.jsonl" 

records = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        record = json.loads(line.strip())  # Parse each line as JSON
        if 'model_input' in record and 'id' in record:  # Ensure keys exist
            records.append({'id': record['id'], 'question': record['model_input'], 'model_id': record['model_id']})  # Extract 'id' and 'model_input'


# Display the contents of the second sheet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_names = [
    "SeaLLMs/SeaLLM-7B-v2.5",
    "openchat/openchat-3.5-0106-gemma",
    "arcee-ai/Arcee-Spark",
]

split_text_array = ["\n<|im_start|>assistant\n", "\nassistant", "\nassistant"]
configs = [
    ("k50_p0.90_t0.1", dict(top_k=50, top_p=0.90, temperature=0.1)),
    ('k50_p0.95_t0.1', dict(top_k=50, top_p=0.95, temperature=0.1)),
    ("k50_p0.90_t0.2", dict(top_k=50, top_p=0.90, temperature=0.2)),
    ('k50_p0.95_t0.2', dict(top_k=50, top_p=0.95, temperature=0.2)),
    ("k50_p0.90_t0.3", dict(top_k=50, top_p=0.90, temperature=0.3)),
    ('k50_p0.95_t0.3', dict(top_k=50, top_p=0.95, temperature=0.3)),
    ("k75_p0.90_t0.1", dict(top_k=75, top_p=0.90, temperature=0.1)),
    ('k75_p0.95_t0.1', dict(top_k=75, top_p=0.95, temperature=0.1)),
    ("k75_p0.90_t0.2", dict(top_k=75, top_p=0.90, temperature=0.2)),
    ('k75_p0.95_t0.2', dict(top_k=75, top_p=0.95, temperature=0.2)),
    ("k75_p0.90_t0.3", dict(top_k=75, top_p=0.90, temperature=0.3)),
    ('k75_p0.95_t0.3', dict(top_k=75, top_p=0.95, temperature=0.3)),
]

if args.model_name == '0':
    model_name = model_names[0]
    split_text = split_text_array[0]

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='balanced', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
    for shorthand, config in tqdm.tqdm(configs):
        
        output_file_path = os.path.join(f'./data/alt_res_{args.data_split}_setv2/alt_res/ar', f'arabic-{model_name.split("/")[1]}.{shorthand}.jsonl')
        os.makedirs(f'./data/alt_res_{args.data_split}_setv2/alt_res/ar', exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as file:

            print("Model used ", model_name)
            model_short = model_name.split("/")[-1]

            for record in tqdm.tqdm(records):

                if record['model_id'] != model_name:  # Skip if model_id doesn't match
                    continue
                        
                id = record['id']
                message = record['question']
                prompt = message


                messages = [
                    {"role": "user", "content": "أجب عن السؤال التالي بشكل دقيق ومختصر"},
                    {
                        "role": "assistant",
                        "content": "بالطبع! ما هو السؤال الذي تود الإجابة عنه؟",
                    },
                    {"role": "user", "content": record['question']},
                ]
                encodeds = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True
                )

                model_inputs = encodeds

                outputs = model.generate(
                    model_inputs,
                    max_new_tokens=512,
                    num_return_sequences=5,
                    no_repeat_ngram_size=2,
                    return_dict_in_generate=True,
                    output_logits=True,
                    do_sample=True,
                    **config,
                )

                for response_index in range(5):
                    response_text = tokenizer.decode(
                        outputs.sequences[response_index], skip_special_tokens=True
                    ).strip()
                    response_text = response_text.split(split_text)[-1].strip()
                    print(response_text)
                    response_token_ids = (
                        outputs.sequences[response_index].to("cpu").tolist()[len(model_inputs[0]) :]
                    )
                    response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)

                    # Extract only the logits corresponding to the output tokens
                    response_logits = [l[response_index, token_id].item() for l, token_id in zip(outputs.logits, response_token_ids)]

                    assert len(response_tokens) == len(response_logits), f"Length mismatch: {len(response_tokens)} != {len(response_logits)}"

                    response_record = {
                            "id": id,
                            "model_id": model_name,
                            "lang": "AR",
                            "response_index": response_index,  # Index of the current response
                            "output_text": response_text,
                            "output_tokens": response_tokens,
                            "output_logits": response_logits,
                        }
                    
                    json.dump(response_record, file, ensure_ascii=False)
                    file.write("\n")

    del model

    torch.cuda.empty_cache()


elif args.model_name == '1':
    model_name = model_names[1]
    split_text = split_text_array[1]

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='balanced', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
    for shorthand, config in tqdm.tqdm(configs):
        
        output_file_path = os.path.join(f'./data/alt_res_{args.data_split}_setv2/alt_res/ar', f'arabic-{model_name.split("/")[1]}.{shorthand}.jsonl')
        os.makedirs(f'./data/alt_res_{args.data_split}_setv2/alt_res/ar', exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as file:

            print("Model used ", model_name)
            model_short = model_name.split("/")[-1]

            for record in tqdm.tqdm(records):

                if record['model_id'] != model_name:  # Skip if model_id doesn't match
                            continue
                        
                id = record['id']
                message = record['question']
                prompt = message


                messages = [
                    {"role": "user", "content": "أجب عن السؤال التالي بشكل دقيق ومختصر"},
                    {
                        "role": "assistant",
                        "content": "بالطبع! ما هو السؤال الذي تود الإجابة عنه؟",
                    },
                    {"role": "user", "content": record['question']},
                ]
                encodeds = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True
                )

                model_inputs = encodeds

                outputs = model.generate(
                    model_inputs,
                    max_new_tokens=512,
                    num_return_sequences=5,
                    no_repeat_ngram_size=2,
                    return_dict_in_generate=True,
                    output_logits=True,
                    do_sample=True,
                    **config,
                )

                for response_index in range(5):
                    response_text = tokenizer.decode(
                        outputs.sequences[response_index], skip_special_tokens=True
                    ).strip()
                    response_text = response_text.split('model\n')[-1].strip() if 'model\n' in response_text else response_text.split(split_text)[-1].strip()
                    print(response_text)
                    response_token_ids = (
                        outputs.sequences[response_index].to("cpu").tolist()[len(model_inputs[0]) :]
                    )
                    response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)

                    # Extract only the logits corresponding to the output tokens
                    response_logits = [l[response_index, token_id].item() for l, token_id in zip(outputs.logits, response_token_ids)]

                    assert len(response_tokens) == len(response_logits), f"Length mismatch: {len(response_tokens)} != {len(response_logits)}"

                    response_record = {
                            "id": id,
                            "model_id": model_name,
                            "lang": "AR",
                            "response_index": response_index,  # Index of the current response
                            "output_text": response_text,
                            "output_tokens": response_tokens,
                            "output_logits": response_logits,
                        }
                    
                    json.dump(response_record, file, ensure_ascii=False)
                    file.write("\n")

    del model
    torch.cuda.empty_cache() 

elif args.model_name == '2':
    model_name = model_names[2]
    split_text = split_text_array[2]

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='balanced', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
    for shorthand, config in tqdm.tqdm(configs):
        
        output_file_path = os.path.join(f'./data/alt_res_{args.data_split}_setv2/alt_res/ar', f'arabic-{model_name.split("/")[1]}.{shorthand}.jsonl')
        os.makedirs(f'./data/alt_res_{args.data_split}_setv2/alt_res/ar', exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as file:

            print("Model used ", model_name)
            model_short = model_name.split("/")[-1]

            for record in tqdm.tqdm(records):

                if record['model_id'] != model_name:  # Skip if model_id doesn't match
                            continue
                        
                id = record['id']
                message = record['question']
                prompt = message


                messages = [
                    {"role": "user", "content": "أجب عن السؤال التالي بشكل دقيق ومختصر"},
                    {
                        "role": "assistant",
                        "content": "بالطبع! ما هو السؤال الذي تود الإجابة عنه؟",
                    },
                    {"role": "user", "content": record['question']},
                ]
                encodeds = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True
                )

                model_inputs = encodeds

                outputs = model.generate(
                    model_inputs,
                    max_new_tokens=512,
                    num_return_sequences=5,
                    no_repeat_ngram_size=2,
                    return_dict_in_generate=True,
                    output_logits=True,
                    do_sample=True,
                    **config,
                )

                for response_index in range(5):
                    response_text = tokenizer.decode(
                        outputs.sequences[response_index], skip_special_tokens=True
                    ).strip()
                    response_text = response_text.split(split_text)[-1].strip()
                    print(response_text)
                    response_token_ids = (
                        outputs.sequences[response_index].to("cpu").tolist()[len(model_inputs[0]) :]
                    )
                    response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)

                    # Extract only the logits corresponding to the output tokens
                    response_logits = [l[response_index, token_id].item() for l, token_id in zip(outputs.logits, response_token_ids)]

                    assert len(response_tokens) == len(response_logits), f"Length mismatch: {len(response_tokens)} != {len(response_logits)}"

                    response_record = {
                            "id": id,
                            "model_id": model_name,
                            "lang": "AR",
                            "response_index": response_index,  # Index of the current response
                            "output_text": response_text,
                            "output_tokens": response_tokens,
                            "output_logits": response_logits,
                        }
                    
                    json.dump(response_record, file, ensure_ascii=False)
                    file.write("\n")
    del model
    torch.cuda.empty_cache()

    os.system('tput bel')
