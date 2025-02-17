import sys
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse as ap
import tqdm as tqdm
import os

# if len(sys.argv) > 1:
#     model_name = sys.argv[1]
# else:
#     model_name = "Qwen/Qwen1.5-14B-Chat"

parser = ap.ArgumentParser()
parser.add_argument("data_split", type=str, help="data split to generate responses for. e.g.: test, val, test_jan25")
parser.add_argument("model_name", type=str, help="model name to use for generation")
args = parser.parse_args()

model_name = args.model_name
if args.data_split == 'test_jan25':
    file_path = f"data/{args.data_split}/mushroom.zh-tst.v1.jsonl"
else:
    file_path = f"data/v2_splits/{args.data_split}/mushroom.zh-val.v2.jsonl" # TODO: change according to needs

records = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        record = json.loads(line.strip())  # Parse each line as JSON
        if 'model_input' in record and 'id' in record:  # Ensure keys exist
            records.append({'id': record['id'], 'question': record['model_input'], 'model_id': record['model_id']})  # Extract 'id' and 'model_input'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)#.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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

for shorthand, config in tqdm.tqdm(configs):

    output_file_path = os.path.join(f'./data/alt_res_{args.data_split}_setv2/alt_res/zh', f'chinese-{model_name.split("/")[1]}.{shorthand}.jsonl')
    os.makedirs(f'./data/alt_res_{args.data_split}_setv2/alt_res/zh', exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as file:
        print("Model used ", model_name)

        for record in records:

            if record['model_id'] != model_name:  # Skip if model_id doesn't match
                continue

            message = record['question']
            if 'internlm' in model_name:
                prompt = f"[INST] {message} [/INST]"
            else:
                prompt = f"{message}"

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            for response_index in range(5):
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=512,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    return_dict_in_generate=True,
                    output_logits=True,
                    do_sample=True, 
                    **config,
                )
                
                id = record['id']
                message = record['question']
                prompt = message

                response_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                if model_name == 'internlm/internlm2-chat-7b':
                    response_text = response_text.split('[/INST]')[-1].strip() # response repeats the input in the begining
                else:
                    response_text = response_text.replace(prompt, '').strip()
                response_token_ids = outputs.sequences[0].to("cpu").tolist()[len(inputs.input_ids[0]):]
                #response_embeddings = [x.to("cpu").tolist() for x in outputs.hidden_states[0][-1][0]] # embedding of the last layer
                response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)
                response_logits = [l[0, token_id].item() for l, token_id in zip(outputs.logits, response_token_ids)]

                assert len(response_tokens) == len(response_logits), f"Length mismatch: {len(response_tokens)} != {len(response_logits)}"

                response_record = {
                        "id": id,
                        "model_id": model_name,
                        "lang": "ZH",
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
