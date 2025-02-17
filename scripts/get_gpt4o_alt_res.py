import os
import json
import random
from pydantic import BaseModel
from openai import OpenAI
from typing import List, Tuple
import itertools
import yaml
from datetime import datetime
import argparse as ap

p = ap.ArgumentParser()
p.add_argument('data_split',type=str, help="test or val or test_jan25 data split")
p.add_argument('--langs', type=str, default=None, help='Choose either one language abbreviation (e.g.: "de")or a selection separated by commas (without whitespace) (e.g.:"en,de,fr"). If argument is not used, all languages will be considered.')
p.add_argument('--corr', action='store_true', help="Used to fill up a 100 paraphrases for GPT4o-based paraphrase generation.")
a = p.parse_args()

if a.langs:
    LANGS = a.langs.split(',')
else:
    LANGS = ['ar', 'en', 'es', 'fi', 'fr', 'hi', 'it', 'sv', 'zh', 'de', 'sv', 'fa', 'eu', 'cs', 'ca']


#org_id = input("Please enter your OpenAI Organization ID:")
#project_id = input("Please enter your OpenAI Project ID:")
#api_key = input("Please enter your OpenAI API key:") 
with open('./.hf/openai_key', 'r') as file:  
    access_token = file.read().strip()
client = OpenAI(api_key=access_token)


class AlternativeAnswer(BaseModel):
    alternative_answer: str

class AlternativeAnswerParaphrases(BaseModel):
    alternative_answer: List[str]

def get_alt_res(lang_code):
    """
    Outputs alternative answers to the hallucinated answer's questions.
    """
    lang_dict = {'ar': 'arabic', 'en': 'english', 'es': 'spanish', 'fi': 'finnish', 'fr': 'french', 'hi': 'hindi', 'it': 'italian', 'sv': 'swedish', 'zh': 'chinese', 'de': 'german', 'fa': 'farsi', 'eu': 'basque', 'cs': 'czech', 'ca': 'catalan'}
    for shorthand, config in configs:
        os.makedirs(f'./data/alt_res_{a.data_split}_setv2/gpt4o/{lang_code}/', exist_ok=True) # TODO: adjust
        if a.data_split == 'test_jan25':
            input_file_path = f"./data/{a.data_split}/mushroom.{lang_code}-tst.v1.jsonl"
        else:
            input_file_path = f"./data/v2_splits/{a.data_split}/mushroom.{lang_code}-val.v2.jsonl"
        output_file_path = f"./data/alt_res_{a.data_split}_setv2/gpt4o/{lang_code}/{lang_dict[lang_code]}-gpt4o.{shorthand}-val.v2.jsonl" # TODO: adjust
        with open(input_file_path, "r", encoding="utf-8") as i, \
            open(output_file_path, "w", encoding="utf-8") as o:
            for line in i:
                sample = json.loads(line)
                pred = {k: sample[k] for k in itertools.islice(sample, 5)}
                model_input = sample['model_input']
                
                mess = [
                            {"role": "system", "content": "You are a helpful AI assistant. Answer the following question or complete the task to the best of your ability. Keep your answer relevant to the question or task and don't include any irrelevant information."},
                            {"role": "user", "content": model_input},
                        ]
                try:
                    completion = client.beta.chat.completions.parse(model="gpt-4o-mini-2024-07-18",
                                                                    messages=mess,
                                                                    response_format=AlternativeAnswer,
                                                                    n=20,
                                                                    seed=42, 
                                                                    max_tokens=1000,
                                                                    **config)
                    output_text_spans = []
                except:
                    try:
                        completion = client.beta.chat.completions.parse(model="gpt-4o-mini-2024-07-18",
                                                                        messages=mess,
                                                                        response_format=AlternativeAnswer,
                                                                        n=20,
                                                                        seed=42, 
                                                                        max_tokens=5000,
                                                                    **config)
                        output_text_spans = []
                    except:
                        print(f"Error in {lang_code} for {sample['id']}")
                        os.makedirs(f'./data/alt_res_{a.data_split}_setv2/gpt4o/{lang_code}/errors', exist_ok=True) 
                        output_text_spans = ['Error']
                for response_index in range(20):
                    if output_text_spans != ['Error']:
                        alternative_answer = completion.choices[response_index].message.parsed.alternative_answer
                        print(response_index, alternative_answer)
                        pred['output_text'] = alternative_answer
                        pred['response_index'] = response_index
                        updated_json = json.dumps(pred)
                        o.write(updated_json + "\n")
                    else:
                        pred['output_text'] = 'Error'
                        pred['response_index'] = response_index
                        updated_json = json.dumps(pred)
                        o.write(updated_json + "\n")
            print(f'\nThe file has been stored under this path:\n{output_file_path}') # TODO: adjust

def get_alt_res_paraphrases(lang_code):
    """
    Outputs alternative answers to the hallucinated answer's questions and parsaphrases them 99 times.
    """
    lang_dict = {'ar': 'arabic', 'en': 'english', 'es': 'spanish', 'fi': 'finnish', 'fr': 'french', 'hi': 'hindi', 'it': 'italian', 'sv': 'swedish', 'zh': 'chinese', 'de': 'german'}
    os.makedirs(f'./data/alt_res_{a.data_split}_setv2/gpt4o-para/{lang_code}/', exist_ok=True) # TODO: adjust
    input_file_path = f"./data/v2_splits/{a.data_split}/mushroom.{lang_code}-val.v2.jsonl"
    output_file_path = f"./data/alt_res_{a.data_split}_setv2/gpt4o-para/{lang_code}/{lang_dict[lang_code]}-gpt4o-para-val.v2.jsonl" # TODO: adjust
    with open(input_file_path, "r", encoding="utf-8") as i, \
        open(output_file_path, "w", encoding="utf-8") as o:
        for line in i:
            sample = json.loads(line)
            pred = {k: sample[k] for k in itertools.islice(sample, 5)}
            model_input = sample['model_input']
            
            num_paras = 0
            aa = []

            while num_paras < 100:
                mess = [
                            {"role": "system", "content": "You are a helpful AI assistant. Answer the following question or complete the task to the best of your ability. Keep your answer relevant to the question or task and don't include any irrelevant information. Then paraphrase the answer 99 times and output all 100 answers in a python list of strings. For example: What is the capital of France?\n['Paris is the capital of France.', 'The French capital is Paris.', 'France is capital city is Paris.', 'In France, the capital is Paris.', 'Paris serves as the capital of France.', 'The main city of France, its capital, is Paris.', 'The administrative center of France is Paris.', 'Paris, known as France is capital.', 'The city of Paris holds the title of France is capital.', 'The capital city of France happens to be Paris.', 'Paris is the capital of France.', 'The French capital is Paris.', 'France is capital city is Paris.', 'In France, the capital is Paris.', 'Paris serves as the capital of France.', 'The main city of France, its capital, is Paris.', 'The administrative center of France is Paris.', 'Paris, known as France is capital.', 'The city of Paris holds the title of France is capital.', 'The capital city of France happens to be Paris.', 'Paris is the capital of France.', 'The French capital is Paris.', 'France is capital city is Paris.', 'In France, the capital is Paris.', 'Paris serves as the capital of France.', 'The main city of France, its capital, is Paris.', 'The administrative center of France is Paris.', 'Paris, known as France is capital.', 'The city of Paris holds the title of France is capital.', 'The capital city of France happens to be Paris.', 'Paris is the capital of France.', 'The French capital is Paris.', 'France is capital city is Paris.', 'In France, the capital is Paris.', 'Paris serves as the capital of France.', 'The main city of France, its capital, is Paris.', 'The administrative center of France is Paris.', 'Paris, known as France is capital.', 'The city of Paris holds the title of France is capital.', 'The capital city of France happens to be Paris.', 'Paris is the capital of France.', 'The French capital is Paris.', 'France is capital city is Paris.', 'In France, the capital is Paris.', 'Paris serves as the capital of France.', 'The main city of France, its capital, is Paris.', 'The administrative center of France is Paris.', 'Paris, known as France is capital.', 'The city of Paris holds the title of France is capital.', 'The capital city of France happens to be Paris.', 'Paris is the capital of France.', 'The French capital is Paris.', 'France is capital city is Paris.', 'In France, the capital is Paris.', 'Paris serves as the capital of France.', 'The main city of France, its capital, is Paris.', 'The administrative center of France is Paris.', 'Paris, known as France is capital.', 'The city of Paris holds the title of France is capital.', 'The capital city of France happens to be Paris.', 'Paris is the capital of France.', 'The French capital is Paris.', 'France is capital city is Paris.', 'In France, the capital is Paris.', 'Paris serves as the capital of France.', 'The main city of France, its capital, is Paris.', 'The administrative center of France is Paris.', 'Paris, known as France is capital.', 'The city of Paris holds the title of France is capital.', 'The capital city of France happens to be Paris.', 'Paris is the capital of France.', 'The French capital is Paris.', 'France is capital city is Paris.', 'In France, the capital is Paris.', 'Paris serves as the capital of France.', 'The main city of France, its capital, is Paris.', 'The administrative center of France is Paris.', 'Paris, known as France is capital.', 'The city of Paris holds the title of France is capital.', 'The capital city of France happens to be Paris.', 'Paris is the capital of France.', 'The French capital is Paris.', 'France is capital city is Paris.', 'In France, the capital is Paris.', 'Paris serves as the capital of France.', 'The main city of France, its capital, is Paris.', 'The administrative center of France is Paris.', 'Paris, known as France is capital.', 'The city of Paris holds the title of France is capital.', 'The capital city of France happens to be Paris.', 'Paris is the capital of France.', 'The French capital is Paris.', 'France is capital city is Paris.', 'In France, the capital is Paris.', 'Paris serves as the capital of France.', 'The main city of France, its capital, is Paris.', 'The administrative center of France is Paris.', 'Paris, known as France is capital.', 'The city of Paris holds the title of France is capital.', 'The capital city of France happens to be Paris.']"},
                            {"role": "user", "content": model_input},
                        ]
                try:
                    completion = client.beta.chat.completions.parse(model="gpt-4o-mini-2024-07-18",
                                                                    messages=mess,
                                                                    response_format=AlternativeAnswerParaphrases,
                                                                    n=1,
                                                                    seed=42, 
                                                                    max_tokens=2000,
                                                                    temperature=0)
                    alternative_answer = completion.choices[0].message.parsed.alternative_answer
                except:
                    try:
                        completion = client.beta.chat.completions.parse(model="gpt-4o-mini-2024-07-18",
                                                                    messages=mess,
                                                                    response_format=AlternativeAnswerParaphrases,
                                                                    n=1,
                                                                    max_tokens=16000,
                                                                    seed=42,
                                                                    temperature=0)
                    except:
                        alternative_answer = ['']
                        print(0, alternative_answer)
                        print(len(alternative_answer))
                        aa.extend(alternative_answer)
                        num_paras += len(alternative_answer)
                        continue
                    
                    alternative_answer = completion.choices[0].message.parsed.alternative_answer

                print(0, alternative_answer)
                print(len(alternative_answer))
                aa.extend(alternative_answer)
                num_paras += len(alternative_answer)
            
            assert len(aa) >= 100

            print(len(aa))
            for alt_res, response_index in zip(aa[:100], range(len(aa[:100]))):
                pred['output_text'] = alt_res
                pred['response_index'] = response_index
                pred['num_paraphrases'] = len(alternative_answer)
                updated_json = json.dumps(pred)
                o.write(updated_json + "\n")
        print(f'\nThe file has been stored under this path:\n{output_file_path}') # TODO: adjust


def get_missing_paras(lang_code):
    lang_dict = {'ar': 'arabic', 'en': 'english', 'es': 'spanish', 'fi': 'finnish', 'fr': 'french', 'hi': 'hindi', 'it': 'italian', 'sv': 'swedish', 'zh': 'chinese', 'de': 'german'}
    org_output_file_path = f"./data/alt_res_{a.data_split}_setv2/gpt4o-para/{lang_code}/{lang_dict[lang_code]}-gpt4o-para-val.v2.jsonl"

    with open(org_output_file_path, "r", encoding="utf-8") as org_input:
        org_data = [json.loads(line) for line in org_input]
    
    unique_samples = {sample['id']: sample for sample in org_data}

    samples_w_less_than_100_paras = [sample for sample in unique_samples if sample['num_paraphrases'] < 100]

    for sample in samples_w_less_than_100_paras:
        pred = {k: sample[k] for k in itertools.islice(sample, 5)}
        model_input = sample['model_input']
        num_paras = sample['num_paraphrases']

        while num_paras < 100:
            mess = [
                        {"role": "system", "content": "You are a helpful AI assistant. Answer the following question or complete the task to the best of your ability. Keep your answer relevant to the question or task and don't include any irrelevant information. Then paraphrase the answer 99 times and output all 100 answers in a python list of strings. For example: What is the capital of France?\n['Paris is the capital of France.', 'The French capital is Paris.', 'France is capital city is Paris.', 'In France, the capital is Paris.', 'Paris serves as the capital of France.', 'The main city of France, its capital, is Paris.', 'The administrative center of France is Paris.', 'Paris, known as France is capital.', 'The city of Paris holds the title of France is capital.', 'The capital city of France happens to be Paris.', 'Paris is the capital of France.', 'The French capital is Paris.', 'France is capital city is Paris.', 'In France, the capital is Paris.', 'Paris serves as the capital of France.', 'The main city of France, its capital, is Paris.', 'The administrative center of France is Paris.', 'Paris, known as France is capital.', 'The city of Paris holds the title of France is capital.', 'The capital city of France happens to be Paris.', 'Paris is the capital of France.', 'The French capital is Paris.', 'France is capital city is Paris.', 'In France, the capital is Paris.', 'Paris serves as the capital of France.', 'The main city of France, its capital, is Paris.', 'The administrative center of France is Paris.', 'Paris, known as France is capital.', 'The city of Paris holds the title of France is capital.', 'The capital city of France happens to be Paris.', 'Paris is the capital of France.', 'The French capital is Paris.', 'France is capital city is Paris.', 'In France, the capital is Paris.', 'Paris serves as the capital of France.', 'The main city of France, its capital, is Paris.', 'The administrative center of France is Paris.', 'Paris, known as France is capital.', 'The city of Paris holds the title of France is capital.', 'The capital city of France happens to be Paris.', 'Paris is the capital of France.', 'The French capital is Paris.', 'France is capital city is Paris.', 'In France, the capital is Paris.', 'Paris serves as the capital of France.', 'The main city of France, its capital, is Paris.', 'The administrative center of France is Paris.', 'Paris, known as France is capital.', 'The city of Paris holds the title of France is capital.', 'The capital city of France happens to be Paris.', 'Paris is the capital of France.', 'The French capital is Paris.', 'France is capital city is Paris.', 'In France, the capital is Paris.', 'Paris serves as the capital of France.', 'The main city of France, its capital, is Paris.', 'The administrative center of France is Paris.', 'Paris, known as France is capital.', 'The city of Paris holds the title of France is capital.', 'The capital city of France happens to be Paris.', 'Paris is the capital of France.', 'The French capital is Paris.', 'France is capital city is Paris.', 'In France, the capital is Paris.', 'Paris serves as the capital of France.', 'The main city of France, its capital, is Paris.', 'The administrative center of France is Paris.', 'Paris, known as France is capital.', 'The city of Paris holds the title of France is capital.', 'The capital city of France happens to be Paris.', 'Paris is the capital of France.', 'The French capital is Paris.', 'France is capital city is Paris.', 'In France, the capital is Paris.', 'Paris serves as the capital of France.', 'The main city of France, its capital, is Paris.', 'The administrative center of France is Paris.', 'Paris, known as France is capital.', 'The city of Paris holds the title of France is capital.', 'The capital city of France happens to be Paris.', 'Paris is the capital of France.', 'The French capital is Paris.', 'France is capital city is Paris.', 'In France, the capital is Paris.', 'Paris serves as the capital of France.', 'The main city of France, its capital, is Paris.', 'The administrative center of France is Paris.', 'Paris, known as France is capital.', 'The city of Paris holds the title of France is capital.', 'The capital city of France happens to be Paris.', 'Paris is the capital of France.', 'The French capital is Paris.', 'France is capital city is Paris.', 'In France, the capital is Paris.', 'Paris serves as the capital of France.', 'The main city of France, its capital, is Paris.', 'The administrative center of France is Paris.', 'Paris, known as France is capital.', 'The city of Paris holds the title of France is capital.', 'The capital city of France happens to be Paris.']"},
                        {"role": "user", "content": model_input},
                    ]
            try:
                completion = client.beta.chat.completions.parse(model="gpt-4o-mini-2024-07-18",
                                                                messages=mess,
                                                                response_format=AlternativeAnswerParaphrases,
                                                                n=1,
                                                                seed=42, 
                                                                max_tokens=2000,
                                                                temperature=0)
                alternative_answer = completion.choices[0].message.parsed.alternative_answer
            except:
                try:
                    completion = client.beta.chat.completions.parse(model="gpt-4o-mini-2024-07-18",
                                                                messages=mess,
                                                                response_format=AlternativeAnswerParaphrases,
                                                                n=1,
                                                                max_tokens=16000,
                                                                seed=42,
                                                                temperature=0)
                except:
                    alternative_answer = ['']
                    num_paras += len(alternative_answer)

            for alt_res, response_index in zip(alternative_answer, range(len(alternative_answer))):
                pred['output_text'] = alt_res
                pred['response_index'] = response_index
                pred['num_paraphrases'] = len(alternative_answer)
                updated_json = json.dumps(pred)
                with open(org_output_file_path, "a", encoding="utf-8") as o:
                    o.write(updated_json + "\n")
               


if __name__ == "__main__":

    LANGS = a.langs.split(',') if a.langs else LANGS

    for lang in LANGS:
        if lang in ['en', 'fr', 'sv']:
            configs = [
                ("k50_p0.90_t0.1", dict(top_p=0.90, temperature=0.1)),
                ('k50_p0.95_t0.1', dict(top_p=0.95, temperature=0.1)),
                ("k50_p0.90_t0.2", dict(top_p=0.90, temperature=0.2)),
                ('k50_p0.95_t0.2', dict(top_p=0.95, temperature=0.2)),
                ("k50_p0.90_t0.3", dict(top_p=0.90, temperature=0.3)),
                ('k50_p0.95_t0.3', dict(top_p=0.95, temperature=0.3)),
            ]
        elif lang == 'fi':
            configs = [
                ('k50_p0.95_t0.1', dict(top_p=0.95, temperature=0.1)),
                ("k50_p0.90_t0.2", dict(top_p=0.90, temperature=0.2)),
                ('k50_p0.95_t0.2', dict(top_p=0.95, temperature=0.2)),
                ("k50_p0.90_t0.3", dict(top_p=0.90, temperature=0.3)),
                ('k50_p0.95_t0.3', dict(top_p=0.95, temperature=0.3)),
            ]
        elif lang == 'zh':
            configs = [
                ('k50_p0.95_t0.1', dict(top_p=0.95, temperature=0.1)),
                ("k50_p0.90_t0.2", dict(top_p=0.90, temperature=0.2)),
                ('k50_p0.95_t0.2', dict(top_p=0.95, temperature=0.2)),
                ("k50_p0.90_t0.3", dict(top_p=0.90, temperature=0.3)),
                ('k50_p0.95_t0.3', dict(top_p=0.95, temperature=0.3)),
            ]
        elif lang == 'de'   :
            configs = [
                ("k50_p0.90_t0.2", dict(top_p=0.90, temperature=0.2)),
                ("k50_p0.90_t0.3", dict(top_p=0.90, temperature=0.3)),
                ('k50_p0.95_t0.3', dict(top_p=0.95, temperature=0.3)),
            ]
        elif lang == 'hi':
            configs = [
                ("k50_p0.90_t0.3", dict(top_p=0.90, temperature=0.3)),
                ('k50_p0.95_t0.3', dict(top_p=0.95, temperature=0.3)),
            ]
        elif lang == 'it':
            configs = [
                ("k50_p0.90_t0.1", dict(top_p=0.90, temperature=0.1)),
            ]

        get_alt_res(lang)
        #get_alt_res_paraphrases(lang)
        #if a.corr:
        #    get_missing_paras(lang)
    print("Done")
    os.system('tput bel')