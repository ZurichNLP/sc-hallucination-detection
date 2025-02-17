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


LANGS = ['ar', 'en', 'es', 'fi', 'fr', 'hi', 'it', 'sv', 'zh', 'de', 'sv']
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


#org_id = input("Please enter your OpenAI Organization ID:")
#project_id = input("Please enter your OpenAI Project ID:")
#api_key = input("Please enter your OpenAI API key:") 
with open('./.hf/openai_key', 'r', encoding='utf-8') as f:
    api_key = f.read().strip()
client = OpenAI(api_key=api_key)


class Hallucination(BaseModel):
    hallucination_spans: List[str]

class Span(BaseModel):
    text: str
    is_hallucination: bool

class HallucinationConstrained(BaseModel):
    hallucination_spans: List[Span]

class AlternativeAnswer(BaseModel):
    alternative_answer: str


def post_processing_cd(hard_labels):
    """
    Post-processing for hallucination span hard labels as retrieved by constrained decoding.
    """
    if not hard_labels:
        return []
    hard_labels.sort(key=lambda x: x[0])
    merged_spans = [hard_labels[0]] 
    for current_start, current_end in hard_labels[1:]:
        last_start, last_end = merged_spans[-1]
        if current_start == last_end + 1:
            merged_spans[-1] = (last_start, current_end)
        else:
            merged_spans.append((current_start, current_end))
    return merged_spans


def get_hallu_span(lang_code, prompt_type, prompt, cd=False):
    """
    Outputs hallucination spans.
    """
    output_format_hall_span = Hallucination if not cd else HallucinationConstrained
    if split != 'test_jan25':
        os.makedirs(f'./preds_val/{split}/gpt4o/{prompt_type}/', exist_ok=True) 
        input_file_path = f"./data/v2_splits/{split}/mushroom.{lang_code}-val.v2.jsonl"
        output_file_path = f"./preds_v2/{split}/gpt4o/{prompt_type}/mushroom.{lang_code}-val.v2.jsonl" 

    else:
        os.makedirs(f'./preds_test_jan25/gpt4o/{prompt_type}/', exist_ok=True) 
        input_file_path = f"./data/test_jan25/mushroom.{lang_code}-tst.v1.jsonl"
        output_file_path = f"./preds_test_jan25/gpt4o/{prompt_type}/mushroom.{lang_code}-tst.v1.jsonl" 
    with open(input_file_path, "r", encoding="utf-8") as i, \
        open(output_file_path, "w", encoding="utf-8") as o:
        for line in i:
            sample = json.loads(line)
            pred = {k: sample[k] for k in itertools.islice(sample, 5)}
            model_input = sample['model_input']
            model_output = sample['model_output_text']
            if cd:
                print("CONSTRAINED DECODING")
                json_schema = {
                    'name': 'Hallucination',
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'hallucination_spans': {
                                'type': 'array',
                                'description': 'List of text spans, each representing a word from an LM generated text.',
                                'items': {
                                    "type": "object",
                                    "properties": {
                                        "model_output_word": {
                                            "type": "string",
                                            "description": "The actual word in the text span.",
                                            "enum": [word.replace('"', '&quot;') for word in model_output.split()],
                                        },
                                    },
                                    "required": ["model_output_word"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        'required': ['hallucination_spans'],
                        'additionalProperties': False
                    },
                    'strict': True
                }
            
            if not prompt_type.startswith('two-step'):
                
                mess = [
                            {"role": "system", "content": prompt["system_prompt"]},
                            {"role": "user", "content": prompt["user_prompt"].format(model_input=model_input, model_output=model_output)},
                        ]
            else:
                print("TWO STEP")
                mess1 = [
                    {"role": "user", "content": prompt["user_prompt_altanswer"].format(model_input=model_input)},
                ]      
            if cd and 'two-step' not in prompt_type:
                print("CONSTRAINED DECODING")
                response_format = {'type': 'json_schema', 'json_schema': json_schema}
            elif not prompt_type.startswith('two-step'):
                response_format = output_format_hall_span
            else:
                print("TWO STEP")
                response_format = AlternativeAnswer
            try:
                completion = client.beta.chat.completions.parse(model="gpt-4o-mini-2024-07-18",
                                                            temperature= 1 if not prompt_type.startswith('two-step') else 0,
                                                            messages=mess if not prompt_type.startswith('two-step') else mess1,
                                                            response_format=response_format,
                                                            seed=42
                                                            )
                output_text_spans = []
                #print(completion)
            except:
                print(f"Error in {lang_code} for {sample['id']}")
                if split != 'test_jan25':
                    error_file_path = f"./preds_v2/val/gpt4o/{prompt_type}/mushroom.{lang_code}-val.v2.errors.txt"
                else:
                    error_file_path = f"./preds_test_jan25/gpt4o/{prompt_type}/mushroom.{lang_code}-tst.v1.errors.jsonl" 
                output_text_spans = ['Error']
            if not prompt_type.startswith('two-step'):
                if not cd:
                    output_text_spans = completion.choices[0].message.parsed.hallucination_spans if output_text_spans != ['Error'] else []
                else :
                    try:
                        output_text_spans = [_['model_output_word'].replace('&quot;', '"') for _ in json.loads(completion.choices[0].message.content)['hallucination_spans']] if output_text_spans != ['Error'] else []
                        print(completion.choices[0].message.content)
                    except:
                        print(f"Error in {lang_code} for {sample['id']}")
                        error_file_path = f"./preds_v2/val/gpt4o/{prompt_type}/mushroom.{lang_code}-val.v2.errors.txt" # TODO: adjust
                        with open(error_file_path, "a", encoding="utf-8") as e:
                            e.write(f"Error in {lang_code} for {sample['id']}: {completion.choices[0].message.content}")
                        #continue
                        output_text_spans = model_output.split()
                print(output_text_spans)
            else:
                try:
                    alternative_answer = completion.choices[0].message.parsed.alternative_answer
                except:
                    alternative_answer = ""
                mess2 = [
                    {"role": "system", "content": prompt["system_prompt"]},
                    {"role": "user", "content": prompt["user_prompt"].format(model_input=model_input, model_output=model_output, alternative_answer=alternative_answer)},
                ]
                if not cd:
                    response_format = output_format_hall_span
                else:
                    response_format = {'type': 'json_schema', 'json_schema': json_schema}
                try:
                    completion = client.beta.chat.completions.parse(model="gpt-4o-mini-2024-07-18",
                                                            temperature=0, 
                                                            messages=mess2,
                                                            response_format=response_format,
                                                            seed=42
                                                            )
                except:
                    print(f"Error in {lang_code} with for {sample['id']}")
                    if split != 'test_jan25':
                        error_file_path = f"./preds_v2/val/gpt4o/{prompt_type}/mushroom.{lang_code}-val.v2.errors.txt"
                    else:
                        error_file_path = f"./preds_{split}/gpt4o/{prompt_type}/mushroom.{lang_code}-tst.v1.errors.txt"
                    output_text_spans = model_output.split()
                if not cd:
                    output_text_spans = completion.choices[0].message.parsed.hallucination_spans if output_text_spans != [model_output.split()] else []
                else:
                    try:
                        output_text_spans = [_['model_output_word'].replace('&quot;', '"') for _ in json.loads(completion.choices[0].message.content)['hallucination_spans']] if output_text_spans != ['Error'] else []
                        print(completion.choices[0].message.content)
                    except:
                        print(f"Error in {lang_code} with for {sample['id']}")
                        if split != 'test_jan25':
                            error_file_path = f"./preds_v2/val/gpt4o/{prompt_type}/mushroom.{lang_code}-val.v2.errors.txt" # TODO: adjust
                        else:
                            error_file_path = f"./preds_{split}/gpt4o/{prompt_type}/mushroom.{lang_code}-tst.v1.errors.txt"
                        with open(error_file_path, "a", encoding="utf-8") as e:
                            e.write(f"Error in {lang_code} with for {sample['id']}")
                        output_text_spans = model_output.split()
                print(output_text_spans)
                """print('\n')
                print(output_text_spans)"""
            pred['pred_hallu_span'] = output_text_spans 
            hard_labels = get_hard_labels(output_text_spans, model_output, lang_code, prompt_type, cd=True if cd else False) if not cd else post_processing_cd(get_hard_labels(output_text_spans, model_output, lang_code, prompt_type, cd=True if cd else False))
            pred['hard_labels'] = hard_labels
            print(f"GPT-4o-mini: {output_text_spans}")
            print(f"pred: {hard_labels}")
            print('================================================================================')
            pred['soft_labels'] = [{"start":start, "prob": 1.0, "end":end} for start, end in hard_labels] # TODO: actually calculate a real prob – now i only have the dummy value 1
            updated_json = json.dumps(pred)
            o.write(updated_json + "\n")
        print(f'\nThe file has been stored under this path:\n{output_file_path}') # TODO: adjust


def get_hard_labels(output_text_spans, model_output, lang_code, prompt, cd=False):
    hard_labels = []
    current_pos = 0 
    
    if split != 'test_jan25':
        error_file_path = f"./preds_v2/val/gpt4o/{prompt}/mushroom.{lang_code}-val.v2.errors.txt" # TODO: adjust
    else:
        error_file_path = f"./preds_{split}/gpt4o/{prompt}/mushroom.{lang_code}-tst.v1.errors.txt"

    with open(error_file_path, "a", encoding="utf-8") as e:
        """if not cd:"""
        for span in output_text_spans:
            start_idx = model_output.find(span, current_pos)
            if start_idx == -1:
                warning = f"Warning: Span '{span}' not found in original text starting from position {current_pos}"
                print(warning)
                e.write(warning)
                continue

            end_idx = start_idx + len(span)
            if start_idx == end_idx:
                end_idx += 1
            hard_labels.append([start_idx, end_idx])
            current_pos = end_idx
        """else:
            print(output_text_spans)
            for span in output_text_spans:
                if not span.is_hallucination:
                    continue
                start_idx = model_output.find(span.text, current_pos)
                if start_idx == -1:
                    warning = f"Warning: Span '{span.text}' not found in original text starting from position {current_pos}"
                    print(warning)
                    e.write(warning)
                    continue

                end_idx = start_idx + len(span.text)
                hard_labels.append([start_idx, end_idx])
                current_pos = end_idx        """

        return hard_labels


if __name__ == "__main__":
    p = ap.ArgumentParser()
    p.add_argument('prompt',type=str, help="Hallucination detection gpt4o prompt (e.g.: direct,direct2,cot,cot2,cot3,two-step,two-step-multi,two-step-comp,two-step-comp-switch) or 'all' to do all prompts in the file prompts.yaml.")
    p.add_argument('split',type=str, help="Split to use (e.g.: val, test, test_jan25)")
    p.add_argument('--langs', type=str, default=None, help='Choose either one language abbreviation (e.g.: "de")or a selection separated by commas (without whitespace) (e.g.:"en,de,fr"). If argument is not used, all languages will be considered.')
    a = p.parse_args()

    # make split global variable
    split = a.split

    if split == 'test_jan25':
        LANGS = ['ar', 'en', 'es', 'fi', 'fr', 'hi', 'it', 'sv', 'zh', 'de', 'sv', 'eu', 'ca', 'cs', 'fa']

    LANGS = a.langs.split(',') if a.langs else LANGS
    prompt_to_test = a.prompt if a.prompt == 'all' else a.prompt.split(',')

    # the following part post-processes the hallucination spans for existing files
    """for lang in LANGS:
        for prompt in prompt_to_test: 
            file_path = f"./preds_v2/val/gpt4o/{prompt}/mushroom.{lang}-val.v2.jsonl"

            # Read existing data
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Update the data
            updated_labels = []
            for line in lines:
                sample = json.loads(line)
                updated_sample = sample.copy()
                
                hard_labels = sample['hard_labels']
                updated_hard_labels = post_processing_cd(hard_labels)
                updated_sample['hard_labels'] = updated_hard_labels
                updated_sample['soft_labels'] = [{"start": start, "prob": 1.0, "end": end} for start, end in updated_hard_labels]
                
                updated_json = json.dumps(updated_sample)
                updated_labels.append(updated_json)

        # Write updated data back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(updated_labels) + "\n")"""


    #prompt_to_test = ['direct', 'direct2', 'cot', 'cot2', 'cot3', 'two-step', 'two-step-multi', 'two-step-comp', 'two-step-comp-switch'] # TODO: adjust to list of prompts if only a subset of prompts are to be tested
    with open('prompts.yaml', 'r', encoding='utf-8') as f:
        prompt_dict = yaml.safe_load(f)
        for prompt in prompt_dict:
            if (prompt_to_test != 'all' and prompt not in prompt_to_test): 
                continue
            print(f"Prompt type: {prompt}")
            for prompt_pieces in prompt_dict[prompt]:
                print(f"{prompt_pieces}: {prompt_dict[prompt][prompt_pieces]}", end='\n\n')
            for lang in LANGS:
                print(f"Language: {lang}")
                cd = True if 'cd' in prompt else False
                get_hallu_span(lang, prompt, prompt_dict[prompt], cd=cd)
