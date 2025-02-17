import json
import argparse as ap
import os

parser = ap.ArgumentParser()
parser.add_argument('--langs', nargs='+')
args = parser.parse_args()

langs = args.langs

for lang in langs:
    for filename in os.listdir(f'data/alt_res_test_jan25_setv2/alt_res/{lang}'):
        out_dir = f'data/alt_res_test_jan25_setv2/alt_res/{lang}_clean'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = os.path.join(out_dir, filename)
        with open(f'data/alt_res_test_jan25_setv2/alt_res/{lang}/{filename}', 'r', encoding='utf-8') as f, open(out_file, 'w', encoding='utf-8') as out:
            for line in f:
                data = json.loads(line.strip())
                if lang == 'cs':
                    if data['model_id'] == "meta-llama/Meta-Llama-3-8B-Instruct":
                        clean_text = data['output_text'].split('assistant\n\n')[-1]
                    elif data['model_id'] == "mistralai/Mistral-7B-Instruct-v0.3":
                        with open(f'data/test_jan25/mushroom.cs-tst.v1.jsonl', 'r', encoding='utf-8') as g:
                            for line in g:
                                org_data = json.loads(line.strip())
                                if org_data['id'] == data['id']:
                                    clean_text = data['output_text'].replace(org_data['model_input'], '')
                                    break

                elif lang == 'fa':
                    if data['model_id'] == "CohereForAI/aya-23-8B":
                        clean_text = data['output_text'].split('<|CHATBOT_TOKEN|>')[-1]
                    elif data['model_id'] == "meta-llama/Meta-Llama-3-8B-Instruct":
                        pass # TODO: Add the clean text to the

                elif lang == 'eu':
                    if data['model_id'] == "google/gemma-7b-it":
                        clean_text = data['output_text'].split('\nmodel\n')[-1]
                    elif data['model_id'] == "meta-llama/Meta-Llama-3-8B-Instruct":
                        clean_text = data['output_text'].split('assistant\n\n')[-1]

                elif lang == 'ca':
                    if data['model_id'] == "meta-llama/Meta-Llama-3-8B-Instruct":
                        clean_text = data['output_text'].split('assistant\n\n')[-1]
                    elif data['model_id'] == "mistralai/Mistral-7B-Instruct-v0.3":
                        clean_text = data['output_text'].replace("Contesta la pregunta següent de manera precisa i concisa, en català. Per descomptat! Quina pregunta t'agradaria respondre?", '')
                        with open(f'data/test_jan25/mushroom.ca-tst.v1.jsonl', 'r', encoding='utf-8') as g:
                            for line in g:
                                org_data = json.loads(line.strip())
                                if org_data['id'] == data['id']:
                                    clean_text = clean_text.replace(org_data['model_input'], '')
                                    break
                        clean_text = clean_text.strip()
                    elif data['model_id'] == "occiglot/occiglot-7b-es-en-instruct":
                        clean_text = data['output_text'].split('<|im_end|> \n<|im_start|> assistant\n')[-1].strip('<|im_end|>')
                
                clean_data = {
                    **data,
                    'output_text': clean_text
                }
                
                json.dump(clean_data, out, ensure_ascii=False)
                out.write('\n')

    print(f'\n\n**Processed {filename} for {lang}!**\n**Please replace the old files with the cleaned ones manually.**\n**You can use "mv -f /path/to/source/* /path/to/destination/" to replace the files and "rm -r path/to/source" to delete the redundant folder.**')