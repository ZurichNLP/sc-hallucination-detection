import os
import json
import argparse
import itertools
from simalign import SentenceAligner
import collections
import numpy as np
from collections import Counter

def preprocess_farsi(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        response_index_dict = {}
        input_samples = [json.loads(line) for line in f]
        # give each sample with the same id a unique rsponse index
        for sample in input_samples:
            if sample['id'] not in response_index_dict:
                response_index_dict[sample['id']] = 1
            else:
                response_index_dict[sample['id']] += 1
            sample['response_index'] = response_index_dict[sample['id']]

    return input_samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get alignments from a JSON file')
    parser.add_argument('input', type=str, help='Path to folder with text to be analyzed for hallucinations')
    parser.add_argument('alt_res', type=str, help='Path to folder with alternative responses. If --avgs is set, this should be the path to the alignments folder.')
    parser.add_argument('method', type=str, default='f', help='Alignment method to use')
    parser.add_argument('--accelerator', type=str, default='gpu') # continue gpu implementation
    parser.add_argument('--avgs', action='store_true')
    parser.add_argument('--medians', action='store_true')
    parser.add_argument('--gpt', action='store_true')
    parser.add_argument('--gpt-paras', action='store_true')
    parser.add_argument('--less-alt-res-params', nargs='+', help='The number of how many alternative response are to be included per sampling method (e.g.: 5 or 10) and sampling method itself (e.g.: k50_p0.90_t0.1 or all).')
    args = parser.parse_args()


    all_matching_methods = {"a": "inter", "m": "mwmf", "i": "itermax", "f": "fwd", "r": "rev"}



    if args.less_alt_res_params: 
        number_samples = int(args.less_alt_res_params[0])
        k, p, t = tuple(args.less_alt_res_params[1].split('_')) if args.less_alt_res_params[1] != 'all' else ('all', 'all', 'all')
        print(k, p, t)
        split_path = args.alt_res.split('/')
        if args.gpt_paras:
            outfolder = os.path.join("/".join(split_path[:-2]), f'sim_avgs_gpt4o-paras_{args.method}_{args.less_alt_res_params[1]}')
        else:
            if p == 'all':
                outfolder = os.path.join("/".join(split_path[:-2]), f'sim_avgs_{args.method}_{args.less_alt_res_params[1]}') if not args.gpt else os.path.join("/".join(split_path[:-2]), f'sim_avgs_gpt4o_{args.method}_{args.less_alt_res_params[1]}')
            else:
                outfolder = os.path.join("/".join(split_path[:-2]), f'sim_avgs_{args.method}_{args.less_alt_res_params[1]}_{args.less_alt_res_params[0]}') if not args.gpt else os.path.join("/".join(split_path[:-2]), f'sim_avgs_gpt4o_{args.method}_{args.less_alt_res_params[1]}_{args.less_alt_res_params[0]}')

        if args.medians:
            outfolder = outfolder + '_median'
        os.makedirs(outfolder, exist_ok=True)
        outfile = args.input.split('/')[-1]
        with open(os.path.join(outfolder, outfile), 'w', encoding='utf-8') as g:
            
            with open(args.input, 'r', encoding='utf-8') as f:
                input_samples = [json.loads(line) for line in f]
            
            for sample in input_samples:
                new_sample = {k: sample[k] for k in itertools.islice(sample, 7)}
                #print(new_sample)

                new_sample['align_scores_all'] = [] # [(org_token, [score1, score2, ...]), (org_token2, [score1, score2, ...]), ...]
                new_sample['avg_align_scores'] = [] # {[org_token, avg_score], ...}

                if p == 'all':
                    number_samples = 1
                    # k is always 50 because it is not adjustable for gpt4o mini
                    all_configs = {
                        'k50_p0.90_t0.1': ['k50', 'p0.90', 't0.1'],
                        'k50_p0.90_t0.2': ['k50', 'p0.90', 't0.2'],
                        'k50_p0.90_t0.3': ['k50', 'p0.90', 't0.3'],
                        'k50_p0.95_t0.1': ['k50', 'p0.95', 't0.1'],
                        'k50_p0.95_t0.2': ['k50', 'p0.95', 't0.2'],
                        'k50_p0.95_t0.3': ['k50', 'p0.95', 't0.3'],
                    }
                    for filename in os.listdir(args.alt_res):
                        count_dict = {}
                        if (all_configs['k50_p0.90_t0.1'][0] in filename and all_configs['k50_p0.90_t0.1'][1] in filename and all_configs['k50_p0.90_t0.1'][2] in filename) \
                            or (all_configs['k50_p0.90_t0.2'][0] in filename and all_configs['k50_p0.90_t0.2'][1] in filename and all_configs['k50_p0.90_t0.2'][2] in filename) \
                                or (all_configs['k50_p0.90_t0.3'][0] in filename and all_configs['k50_p0.90_t0.3'][1] in filename and all_configs['k50_p0.90_t0.3'][2] in filename) \
                                    or (all_configs['k50_p0.95_t0.2'][0] in filename and all_configs['k50_p0.95_t0.2'][1] in filename and all_configs['k50_p0.95_t0.2'][2] in filename) \
                                        or (all_configs['k50_p0.95_t0.1'][0] in filename and all_configs['k50_p0.95_t0.1'][1] in filename and all_configs['k50_p0.95_t0.1'][2] in filename) \
                                            or (all_configs['k50_p0.95_t0.3'][0] in filename and all_configs['k50_p0.95_t0.3'][1] in filename and all_configs['k50_p0.95_t0.3'][2] in filename):
                            print(filename)
                        else:
                            continue
                       
                        align_path = os.path.join(args.alt_res, filename)
                        if '/fa' in args.alt_res:
                            a = preprocess_farsi(align_path)
                        else:
                            with open(align_path, 'r', encoding='utf-8') as a:
                                a = [json.loads(line) for line in a]

                        for align_data in a:
                            if align_data['id'] == sample['id']:
                                if sample['id'] not in count_dict:
                                    count_dict[sample['id']] = 1
                                elif number_samples > count_dict[sample['id']]:
                                    count_dict[sample['id']] += 1
                                else:
                                    continue

                                if not new_sample['align_scores_all']:
                                    new_sample['align_scores_all'] = [(d['org_token'],[d['score']]) for d in align_data['align_scores']]
                                else:
                                    for i, d in enumerate(align_data['align_scores']):
                                        new_sample['align_scores_all'][i][1].append(d['score'])
                                        assert new_sample['align_scores_all'][i][0] == d['org_token']
                                            
                        
                else: # this triggers if not all configs are to be considered
                    for filename in os.listdir(args.alt_res):
                        count_dict = {}
                        if not (k in filename and p in filename and t in filename):
                            continue
                        print(filename)

                        align_path = os.path.join(args.alt_res, filename)
                        if '/fa' in args.alt_res:
                            a = preprocess_farsi(align_path)
                        else:
                            with open(align_path, 'r', encoding='utf-8') as a:
                                a = [json.loads(line) for line in a]

                        for align_data in a:
                            if align_data['id'] == sample['id']:
                                if sample['id'] not in count_dict:
                                    count_dict[sample['id']] = 1
                                elif number_samples > count_dict[sample['id']]:
                                    count_dict[sample['id']] += 1
                                else:
                                    continue

                                if not new_sample['align_scores_all']:
                                    new_sample['align_scores_all'] = [(d['org_token'],[d['score']]) for d in align_data['align_scores']]
                                else:
                                    for i, d in enumerate(align_data['align_scores']):
                                        new_sample['align_scores_all'][i][1].append(d['score'])
                                        assert new_sample['align_scores_all'][i][0] == d['org_token']
                    
                for org_token, scores in new_sample['align_scores_all']:
                    if args.medians:
                        avg_score = np.median(scores)
                    else:
                        avg_score = sum(scores) / len(scores)
                    new_sample['avg_align_scores'].append({
                        'org_token': org_token,
                        'avg_score': avg_score
                    })

                #print(new_sample['align_scores_all'][i][1])
                if p == 'all':
                    if not new_sample['align_scores_all']:
                        print("\nscores are missing...\n")
                    else:
                        print(len(new_sample['align_scores_all'][i][1]))
                        print(new_sample['align_scores_all'])
                        assert len(new_sample['align_scores_all'][i][1]) == 6
                else:
                    print(new_sample['align_scores_all'])
                    if not new_sample['align_scores_all']:
                        print("\nscores are missing...\n")
                    else:
                        assert len(new_sample['align_scores_all'][i][1]) == number_samples
                    #break
                    
                print(len(new_sample['align_scores_all']), len(new_sample['avg_align_scores']), number_samples)
                assert len(new_sample['align_scores_all']) == len(new_sample['avg_align_scores'])


                # Write the average alignment scores to a file
                print(new_sample)
                g.write(json.dumps(new_sample, ensure_ascii=False) + '\n')

        print(f'\nThe file has been stored under this path:\n{os.path.join(outfolder, outfile)}')
                
                
    else:

        if (args.gpt or args.gpt_paras) and not args.avgs:
            myaligner = SentenceAligner('xlmr', token_type='bpe', matching_methods=args.method, device='cuda:0')

            for filename in os.listdir(args.alt_res):
                alt_res = os.path.join(args.alt_res, filename)
                align_file = os.path.join(args.alt_res.replace('gpt4o', f'alignments_gpt4o_{args.method}'), filename.replace('.jsonl', '') + '_alignments.jsonl') if not args.gpt_paras else os.path.join(args.alt_res.replace(f'gpt4o-para', f'alignments_gpt4o-paras_{args.method}'), filename.replace('.jsonl', '') + '_alignments.jsonl')
                os.makedirs(os.path.dirname(align_file), exist_ok=True)

                # Read entire files into memory once
                with open(args.input, 'r', encoding='utf-8') as f:
                    input_samples = [json.loads(line) for line in f]

                if '/fa' in args.alt_res:
                    alt_samples = preprocess_farsi(alt_res)
                else:
                    with open(alt_res, 'r', encoding='utf-8') as g:
                        alt_samples = [json.loads(line) for line in g]

                # Open the output file for writing
                with open(align_file, 'w', encoding='utf-8') as h:
                    for alt_data in alt_samples:
                        for sample in input_samples:
                            if sample['id'] == alt_data['id']:
                                align_data = {k: sample[k] for k in itertools.islice(sample, 5)}
                                # print(align_data)
                                org = sample['model_output_text']
                                alt = alt_data['output_text']
                                align_data['alt_output_text'] = alt
                                align_data['response_index'] = alt_data['response_index']

                                if not alt:
                                    org_tokens = myaligner.embed_loader.tokenizer.tokenize(org)
                                    align_data['align_scores'] = [
                                            {  
                                                #"org_token_idx": org_tokens.index(token),
                                                #"alt_token_idx": None,
                                                "org_token": token,
                                                #"alt_token": None,
                                                "score": 0
                                            }
                                            for token in org_tokens
                                        ]
                                else:
                                    org_tokens = myaligner.embed_loader.tokenizer.tokenize(org)
                                    alt_tokens = myaligner.embed_loader.tokenizer.tokenize(alt)
                                    alignments = myaligner.get_word_aligns_with_scores(org, alt)
                                    assert len(alignments[all_matching_methods[args.method]]) == len(org_tokens)
                                    # for token, align in zip(org_tokens, alignments['inter']):
                                    #     print(token, alt_tokens[align[0][1]] if alt_tokens[align[0][1]] else 'NO TOKN FOUND', align)

                                    align_data['align_scores'] = [
                                        {  
                                            #"org_token_idx": align[0][0],
                                            #"alt_token_idx": align[0][1] if align[0][1] else -1,
                                            "org_token": token,
                                            #"alt_token": alt_tokens[align[0][1]],
                                            "score": float(align[1])
                                        }
                                        for token, align in zip(org_tokens, alignments[all_matching_methods[args.method]])
                                    ]

                                    # print(align_data['align_scores'])
                                    # print(org_tokens)
                                    for token, scores in zip(org_tokens, align_data['align_scores']):
                                        print(token, scores)
                                    
                                    h.write(json.dumps(align_data, ensure_ascii=False) + '\n')

                                for _ in align_data['align_scores']:
                                    print(align_data['id'], align_data['response_index'], _)

                print(f'\nThe file has been stored under this path:\n{align_file}')

        elif not args.avgs:
            myaligner = SentenceAligner('xlmr', token_type='bpe', matching_methods=args.method, device='cuda:0')

            for filename in os.listdir(args.alt_res):
                alt_res = os.path.join(args.alt_res, filename)
                align_file = os.path.join("/".join(args.alt_res.split('/')[:-2]),f'alignments_{args.method}', args.alt_res.split('/')[-1], filename.replace('.jsonl', '') + '_alignments.jsonl')

                os.makedirs(os.path.dirname(align_file), exist_ok=True)

                # Read entire files into memory once
                with open(args.input, 'r', encoding='utf-8') as f:
                    input_samples = [json.loads(line) for line in f]

                if '/fa' in args.alt_res:
                    alt_samples = preprocess_farsi(alt_res)
                else:
                    with open(alt_res, 'r', encoding='utf-8') as g:
                        alt_samples = [json.loads(line) for line in g]

                # Open the output file for writing
                with open(align_file, 'w', encoding='utf-8') as h:
                    for alt_data in alt_samples:
                        for sample in input_samples:
                            if sample['id'] == alt_data['id']:
                                align_data = {k: sample[k] for k in itertools.islice(sample, 5)}
                                # print(align_data)
                                org = sample['model_output_text']
                                alt = alt_data['output_text'] 
                                align_data['alt_output_text'] = alt
                                align_data['response_index'] = alt_data['response_index']

                                if not alt:
                                    org_tokens = myaligner.embed_loader.tokenizer.tokenize(org)
                                    align_data['align_scores'] = [
                                            {  
                                                #"org_token_idx": org_tokens.index(token),
                                                #"alt_token_idx": None,
                                                "org_token": token,
                                                #"alt_token": None,
                                                "score": 0
                                            }
                                            for token in org_tokens
                                        ]
                                else:
                                    org_tokens = myaligner.embed_loader.tokenizer.tokenize(org)
                                    alt_tokens = myaligner.embed_loader.tokenizer.tokenize(alt)
                                    alignments = myaligner.get_word_aligns_with_scores(org, alt)
                                    assert len(alignments[all_matching_methods[args.method]]) == len(org_tokens)
                                    # for token, align in zip(org_tokens, alignments['inter']):
                                    #     print(token, alt_tokens[align[0][1]] if alt_tokens[align[0][1]] else 'NO TOKN FOUND', align)

                                    align_data['align_scores'] = [
                                        {  
                                            #"org_token_idx": align[0][0],
                                            #"alt_token_idx": align[0][1] if align[0][1] else -1,
                                            "org_token": token,
                                            #"alt_token": alt_tokens[align[0][1]],
                                            "score": float(align[1])
                                        }
                                        for token, align in zip(org_tokens, alignments[all_matching_methods[args.method]])
                                    ]

                                    # print(align_data['align_scores'])
                                    # print(org_tokens)
                                    for token, scores in zip(org_tokens, align_data['align_scores']):
                                        print(token, scores)
                                    #print(len(align_data['align_scores']), len(org_tokens))
                                
                                # TODO: comment in in case this works
                                # unique_align_scores = {}
                                # for entry in align_data['align_scores']:
                                #     org_idx = entry['org_token_idx']
                                #     if org_idx not in unique_align_scores or unique_align_scores[org_idx]['score'] < entry['score']:
                                #         unique_align_scores[org_idx] = entry

                                # align_data['align_scores'] = list(unique_align_scores.values())

                                # Write to the output file
                                h.write(json.dumps(align_data, ensure_ascii=False) + '\n')

                                for _ in align_data['align_scores']:
                                    print(align_data['id'], align_data['response_index'], _)

                print(f'\nThe file has been stored under this path:\n{align_file}')
        else:
            split_path = args.alt_res.split('/')
            if args.gpt_paras:
                outfolder = os.path.join("/".join(split_path[:-2]), f'sim_avgs_gpt4o-paras_{args.method}')
            else:
                outfolder = os.path.join("/".join(split_path[:-2]), f'sim_avgs_{args.method}') if not args.gpt else os.path.join("/".join(split_path[:-2]), f'sim_avgs_gpt4o_{args.method}')
            if args.medians:
                outfolder = outfolder + '_median'
            os.makedirs(outfolder, exist_ok=True)
            outfile = args.input.split('/')[-1]
            with open(os.path.join(outfolder, outfile), 'w', encoding='utf-8') as g:

                with open(args.input, 'r', encoding='utf-8') as f:
                        input_samples = [json.loads(line) for line in f]
                
                for sample in input_samples:
                    new_sample = {k: sample[k] for k in itertools.islice(sample, 7)}
                    #print(new_sample)

                    new_sample['align_scores_all'] = [] # [(org_token, [score1, score2, ...]), (org_token2, [score1, score2, ...]), ...]
                    new_sample['avg_align_scores'] = [] # {[org_token, avg_score], ...}

                    for filename in os.listdir(args.alt_res):
                        
                        # Skip files not related to the current sample's `model_id`
                        print(sample['model_id'].split('/')[-1], filename)
                        if not (args.gpt or args.gpt_paras):
                            if sample['model_id'].split('/')[-1] not in filename:
                                continue

                        alt_res = os.path.join(args.alt_res, filename)
                        if '/fa' in args.alt_res:
                            a = preprocess_farsi(alt_res)
                        else:
                            with open(alt_res, 'r', encoding='utf-8') as a:
                                a = [json.loads(line) for line in a]

                        for align_data in a:
                            if align_data['id'] == sample['id']:
                                if not new_sample['align_scores_all']:
                                    new_sample['align_scores_all'] = [(d['org_token'],[d['score']]) for d in align_data['align_scores']]
                                else:
                                    for i, d in enumerate(align_data['align_scores']):
                                        new_sample['align_scores_all'][i][1].append(d['score'])
                                        assert new_sample['align_scores_all'][i][0] == d['org_token']
                
                    for org_token, scores in new_sample['align_scores_all']:
                        if args.medians:
                            avg_score = np.median(scores)
                        else:
                            avg_score = sum(scores) / len(scores)
                        new_sample['avg_align_scores'].append({
                            'org_token': org_token,
                            'avg_score': avg_score
                        })
                    
                    assert len(new_sample['align_scores_all']) == len(new_sample['avg_align_scores'])

                    # Write the average alignment scores to a file
                    print(new_sample)
                    g.write(json.dumps(new_sample, ensure_ascii=False) + '\n')

            print(f'\nThe file has been stored under this path:\n{os.path.join(outfolder, outfile)}')