from huggingface_hub import InferenceClient
import json
import os
import time
import argparse as ap
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_4bit=True)


# read hf token from file
with open('./.hf/token', 'r') as file:
    hf_token = file.read().strip()


parser = ap.ArgumentParser()
parser.add_argument("data_split", type=str, help="data split to generate responses for. e.g.: test, val, test_jan25")
parser.add_argument("model_name", type=str, help="model name to use for generation")
args = parser.parse_args()

model_name = args.model_name

if args.data_split == 'test_jan25':
    file_path = f"data/{args.data_split}/mushroom.sv-tst.v1.jsonl"
else:   
    file_path = f"data/v2_splits/{args.data_split}/mushroom.sv-val.v2.jsonl" # TODO: change according to needs



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if 'gptsw3' in model_name:
    from huggingface_hub import hf_hub_download

    model_name_or_path = "AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct-gguf"
    model_basename = "gpt-sw3-6.7b-v2-instruct-Q4_K_M.gguf"
    model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

    """from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, gguf_file=model_basename, device_map='auto', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, gguf_file=model_basename, trust_remote_code=True) 
    """

    # This config has been tested on an RTX 2080 (VRAM of 11GB).
    # you might need to tweak with respect to your hardware.
    from llama_cpp import Llama
    lcpp_llm = Llama(
        model_path=model_path,
        #n_threads=4, # CPU cores
        n_batch=1024, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        n_gpu_layers=30, # Change this value based on your model and your GPU VRAM pool.
        n_ctx=2048, # Context window
        logits_all=True
    )



    records = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line.strip())  # Parse each line as JSON
            if 'model_input' in record and 'id' in record:  # Ensure keys exist
                records.append({'id': record['id'], 'question': record['model_input'], 'model_id': record['model_id']})  # Extract 'id' and 'model_input'



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
        output_file_path = os.path.join(f'./data/alt_res_{args.data_split}_setv2/alt_res/sv', f'swedish-{model_name_or_path.split("/")[1]}.{shorthand}.jsonl')
        os.makedirs(f'./data/alt_res_{args.data_split}_setv2/alt_res/sv', exist_ok=True)

        with open(output_file_path, 'w') as ostr:
            for record in tqdm.tqdm(records, desc='items'):
                if record['model_id'] != model_name_or_path:
                    continue
                
                id = record['id']
                message = record['question']
                prompt = f"<|endoftext|><s>User:{message}<s>Bot:"

                """inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs.input_ids#.to(model.device)

                terminators = [
                        tokenizer.eos_token_id,
                        tokenizer.encode('\n')[-1],
                    ]

                if 'alt_question' in record:
                    del record['alt_question']

                outputs = model.generate(
                    input_ids,
                    max_new_tokens=512,
                    num_return_sequences=5,
                    eos_token_id=terminators,
                    pad_token_id=tokenizer.pad_token_id,
                    attention_mask=inputs.attention_mask,
                    return_dict_in_generate=True,
                    output_logits=True,
                    do_sample=True,
                    **config
                )"""

                    # print(
                    #     json.dumps(
                    #         {
                    #             **record, 
                    #             'model_output': response['choices'][0]['text'],
                    #             'tokens': response['choices'][0]['logprobs']['tokens'],
                    #             'logprobs': [
                    #                 {k: float(v) for k,v in d.items()} 
                    #                 for d in response['choices'][0]['logprobs']['top_logprobs']
                    #             ],
                    #             'lang': 'DE',
                    #         }
                    #     ), 
                    #     file=ostr_logprobs,
                    # )
                    
                    #print(response['choices'][0]['text'])

                num_responses = 5

                for i in range(num_responses):
                    response = lcpp_llm(
                        prompt=prompt,
                        logprobs=32_000,
                        max_tokens=512,
                        **config
                    )

                print(
                        json.dumps(
                            {
                                **record, 
                                'id': id,
                                'model_id': model_basename,
                                'lang': 'SV',
                                'response_index': i,
                                'output_text': response['choices'][0]['text'].replace(prompt, " ").strip(),
                                'output_tokens': response['choices'][0]['logprobs']['tokens'],
                            }
                        ), 
                        file=ostr,
                        flush=True,
                    )
                
                """for response_index in range(5):
                    response = outputs.sequences[response_index][input_ids.shape[-1]:]
                    response_text = tokenizer.decode(response, skip_special_tokens=True)
                    
                    response_token_ids = response.to("cpu").tolist()
                    response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)
                    response_logits = [l[response_index, token_id].item() for l, token_id in zip(outputs.logits, response_token_ids)]

                    assert len(response_tokens) == len(response_logits), f"Length mismatch: {len(response_tokens)} != {len(response_logits)}"

                response_record = {
                        "id": id,   
                        "model_id": model_name_or_path,
                        "lang": "SV",
                        "response_index": response_index,  # Index of the current response
                        "output_text": response_text,
                        "output_tokens": response_tokens,
                        "output_logits": response_logits,
                }   

                json.dump(response_record, ostr, ensure_ascii=False)
                ostr.write("\n")    """



elif 'viking' in model_name.lower():
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                device_map="auto",
                                                torch_dtype=torch.bfloat16, 
                                                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                                                trust_remote_code=True)#.to(device)
    
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

    records = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line.strip())  # Parse each line as JSON
            if 'model_input' in record and 'id' in record:  # Ensure keys exist
                records.append({'id': record['id'], 'question': record['model_input'], 'model_id': record['model_id']})  # Extract 'id' and 'model_input'

    for shorthand, config in tqdm.tqdm(configs):
        output_file_path = os.path.join(f'./data/alt_res_{args.data_split}_setv2/alt_res/sv', f'swedish-{model_name.split("/")[1]}.{shorthand}.jsonl')
        os.makedirs(f'./data/alt_res_{args.data_split}_setv2/alt_res/sv', exist_ok=True)

        with open(output_file_path, 'w', encoding='utf-8') as file:
            print("Model used ", model_name)

            for record in records:
                
                if record['model_id'] != model_name:  # Skip if model_id doesn't match
                    continue

                prompt = record['question'].rstrip('\n')
                 
                if 'viking' in model_name.lower():
                    system_prompt = "Du är en hjälpsam AI-assistent. Svara på följande fråga eller slutför uppgiften efter bästa förmåga. Håll ditt svar relevant för frågan eller uppgiften och inkludera inte någon irrelevant information. Följande är ett exempel på en lyckad konversation:\nFråga: Vad är Frankrikes huvudstad? Svar: Frankrikes huvudstad är Paris."
                    prompt = f"\nFråga: {prompt} Svar:"
                    prompt = system_prompt + prompt
                
                print('PROMPT', prompt)
                #message = [{"role": "user", "content": message}]
                #prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
                output = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    num_return_sequences=5,
                    return_dict_in_generate=True,
                    #output_scores=True,
                    do_sample=True,
                    #eos_token_id=tokenizer.eos_token_id,
                    #pad_token_id=tokenizer.eos_token_id,
                    #stop_sequences=["User:","\n","Fråga:","Bot:"],
                    **config,
                    )
            

                id = record['id']
                message = record['question']

                for response_index in range(5):
                    response_text = tokenizer.decode(output.sequences[response_index], skip_special_tokens=True)
                    response_text = response_text.replace(prompt, "").split('Svar:')[0].strip().split('Fråga:')[0].strip() # response repeats the input in the begining

                    print('RESPONSE:', response_text)
                    # response_text = response_text.replace(prompt, " ").strip()
                    # Tokenization post-response generation
                    response_token_ids = output.sequences[response_index].to("cpu").tolist()[len(input_ids[0]):]
                    response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)
                        
                    """response_logits = []
                    for i, token_id in enumerate(response_token_ids):
                        token_logits = output.scores[i].to("cpu")[0, token_id].item()
                        response_logits.append(token_logits)"""

                    #assert len(response_tokens) == len(response_logits), f"Length mismatch: {len(response_tokens)} != {len(response_logits)}"

                    response_record = {
                            "id": id,
                            "model_id": model_name,
                            "lang": "SV",
                            "response_index": response_index,  # Index of the current response
                            "output_text": response_text,
                            "output_tokens": response_tokens,
                           # "output_logits": response_logits,
                        }
                    
                    json.dump(response_record, file, ensure_ascii=False)
                    file.write("\n")


else:
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                device_map="auto",
                                                torch_dtype=torch.bfloat16, 
                                                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                                                trust_remote_code=True)#.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    configs = [
        ("k50_p0.90_t0.1", dict(top_k=50, top_p=0.90, temperature=0.1)),
        # ('k50_p0.95_t0.1', dict(top_k=50, top_p=0.95, temperature=0.1)),
        # ("k50_p0.90_t0.2", dict(top_k=50, top_p=0.90, temperature=0.2)),
        # ('k50_p0.95_t0.2', dict(top_k=50, top_p=0.95, temperature=0.2)),
        # ("k50_p0.90_t0.3", dict(top_k=50, top_p=0.90, temperature=0.3)),
        # ('k50_p0.95_t0.3', dict(top_k=50, top_p=0.95, temperature=0.3)),
        # ("k75_p0.90_t0.1", dict(top_k=75, top_p=0.90, temperature=0.1)),
        # ('k75_p0.95_t0.1', dict(top_k=75, top_p=0.95, temperature=0.1)),
        # ("k75_p0.90_t0.2", dict(top_k=75, top_p=0.90, temperature=0.2)),
        # ('k75_p0.95_t0.2', dict(top_k=75, top_p=0.95, temperature=0.2)),
        # ("k75_p0.90_t0.3", dict(top_k=75, top_p=0.90, temperature=0.3)),
        # ('k75_p0.95_t0.3', dict(top_k=75, top_p=0.95, temperature=0.3)),
    ]

    records = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line.strip())  # Parse each line as JSON
            if 'model_input' in record and 'id' in record:  # Ensure keys exist
                records.append({'id': record['id'], 'question': record['model_input'], 'model_id': record['model_id']})  # Extract 'id' and 'model_input'

    for shorthand, config in tqdm.tqdm(configs):
        output_file_path = os.path.join(f'./data/alt_res_{args.data_split}_setv2/alt_res/sv', f'swedish-{model_name.split("/")[1]}.{shorthand}.jsonl')
        os.makedirs(f'./data/alt_res_{args.data_split}_setv2/alt_res/sv', exist_ok=True)

        with open(output_file_path, 'w', encoding='utf-8') as file:
            print("Model used ", model_name)

            for record in records:
                
                if record['model_id'] != model_name:  # Skip if model_id doesn't match
                    continue

                message = record['question'].rstrip('\n')
                prompt = f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant "
                #prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
                output = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    num_return_sequences=5,
                    return_dict_in_generate=True,
                    #output_scores=True,
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
                    response_text = response_text.split('assistant ')[-1] # response repeats the input in the begining

                    print(response_text)
                    # response_text = response_text.replace(prompt, " ").strip()
                    # Tokenization post-response generation
                    response_token_ids = output.sequences[response_index].to("cpu").tolist()[len(input_ids[0]):]
                    response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)
                        
                    """response_logits = []
                    for i, token_id in enumerate(response_token_ids):
                        token_logits = output.scores[i].to("cpu")[0, token_id].item()
                        response_logits.append(token_logits)"""

                    #assert len(response_tokens) == len(response_logits), f"Length mismatch: {len(response_tokens)} != {len(response_logits)}"

                    response_record = {
                            "id": id,
                            "model_id": model_name,
                            "lang": "SV",
                            "response_index": response_index,  # Index of the current response
                            "output_text": response_text,
                            "output_tokens": response_tokens,
                           # "output_logits": response_logits,
                        }
                    
                    json.dump(response_record, file, ensure_ascii=False)
                    file.write("\n")

os.system('tput bel')
print('Script done')