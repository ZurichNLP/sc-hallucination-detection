import json
import csv
import torch
import pandas as pd
import tqdm
import argparse as ap
import os

from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM

set_seed(94326)


parser = ap.ArgumentParser()
parser.add_argument("data_split", type=str, help="data split to generate responses for. e.g.: test, val, test_jan25")

args = parser.parse_args()

if args.data_split == 'test_jan25':
    file_path = f"data/{args.data_split}/mushroom.fr-tst.v1.jsonl"
else:
    file_path = f"data/v2_splits/{args.data_split}/mushroom.fr-val.v2.jsonl" # TODO: change according to needs

with open('./.hf/token', 'r') as file:
    hftoken = file.readlines()[0].strip()

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        trust_remote_code=True,
        cache_dir=".hf",
        token=hftoken,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hftoken,
    )
    return model, tokenizer

def main():
    records = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line.strip())  # Parse each line as JSON
            if 'model_input' in record and 'id' in record:  # Ensure keys exist
                records.append({'id': record['id'], 'question': record['model_input'], 'model_id': record['model_id']})  # Extract 'id' and 'model_input'

    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    model, tokenizer = load_model(model_name)
    configs = [
        ('k50_p0.90_t0.1', dict(top_k=50, top_p=0.90, temperature=0.1)),
        ('k50_p0.90_t0.2', dict(top_k=50, top_p=0.90, temperature=0.2)),
        ('k50_p0.90_t0.3', dict(top_k=50, top_p=0.90, temperature=0.3)),
        ('k50_p0.95_t0.1', dict(top_k=50, top_p=0.95, temperature=0.1)),
        ('k50_p0.95_t0.2', dict(top_k=50, top_p=0.95, temperature=0.2)),
        ('k50_p0.95_t0.3', dict(top_k=50, top_p=0.95, temperature=0.3)),
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
        output_file_path = os.path.join(f'./data/alt_res_{args.data_split}_setv2/alt_res/fr', f'french-{model_name.split("/")[1]}.{shorthand}.jsonl')
        os.makedirs(f'./data/alt_res_{args.data_split}_setv2/alt_res/fr', exist_ok=True)
       
        with open(output_file_path, 'w') as file:
            for record_ in tqdm.tqdm(records, desc=shorthand):
                record = {k: v for k, v in record_.items()}

                if record['model_id'] != model_name:
                    continue

                message = record['question'].strip()
                message = [{"role": "user", "content": message}]
                prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=512,
                    num_return_sequences=5,
                    return_dict_in_generate=True,
                    output_logits=True,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    **config,
                )
                id = record['id']
                message = record['question']
                prompt = message

                for response_index in range(5):
                    # print("output:", output)
                    response_text = tokenizer.decode(outputs.sequences[response_index], skip_special_tokens=False)
                    response_text = response_text.replace(prompt, "", 1) # response repeats the input in the begining
                    response_token_ids = outputs.sequences[response_index].to("cpu").tolist()[len(input_ids[0]):]
                    # response_embeddings = outputs.sequences[0].to("cpu").tolist()[len(inputs.input_ids[0]):]
                    response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)
                    response_text = tokenizer.decode(response_token_ids, skip_special_tokens=True)
                    response_logits = [l[response_index, token_id].item() for l, token_id in zip(outputs.logits, response_token_ids)]
                    assert len(response_tokens) == len(response_logits), f"Length mismatch: {len(response_tokens)} != {len(response_logits)}"

                    #print("prompt:", prompt)
                    #print("response_text:", response_text)
                    #print("response_logits:", len(response_logits))
                    #print("response_tokens:", len(response_tokens))
                    response_record = {
                        "id": id,
                        "model_id": model_name,
                        "lang": "FR",
                        "response_index": response_index,  # Index of the current response
                        "output_text": response_text,
                        "output_tokens": response_tokens,
                        "output_logits": response_logits,
                    }

                    json.dump(response_record, file, ensure_ascii=False)
                    file.write('\n')

if __name__ == "__main__":
    main()

os.system('tput bel')