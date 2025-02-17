import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
import unicodedata
import re
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score
import argparse

parser = argparse.ArgumentParser(description='Get alignments from a JSON file')
parser.add_argument('sim_avg_path', type=str, help='folder containing similarity averages')
parser.add_argument('threshold_metric', type=str, help='which metric should maximized when choosing the threshold, e.g.: f1 or iou')
parser.add_argument('avg_method', type=str, help="choose the averaging method of the similarity scores, e.g.: mean or median")
parser.add_argument('--langs', nargs='+', help='language to be analyzed')
parser.add_argument('--fewer-altres', action='store_true')
parser.add_argument('--full-val', action='store_true')
args = parser.parse_args()
    

# Directories and settings
sim_avgs = args.sim_avg_path

LANGS = ["de", 'en', 'fr', 'ar', 'zh', 'hi', 'it', 'es', 'fi', 'sv'] 

model_data = defaultdict(lambda: {"hallucinated": [], "non_hallucinated": []})

def is_hallucinated(char_start, char_end, hard_labels):
    for start, end in hard_labels:
        if start < char_end and char_start < end:
            return True
    return False

def normalize_text(text):
    return unicodedata.normalize("NFC", text).strip()


# Load existing thresholds from JSON file
thresholds_file = "./scripts/sim_thresholds_by_model.json"
if os.path.exists(thresholds_file):
    with open(thresholds_file, "r", encoding="utf-8") as json_file:
        sim_thresholds = json.load(json_file)
else:
    sim_thresholds = {}

for filename in os.listdir(sim_avgs):
    if args.langs and not any([lang in filename for lang in args.langs]):
        continue
    lang = filename.split('-')[0].split('.')[-1]
    with open(os.path.join(sim_avgs, filename), 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            model_id = sample['model_id']
            print(model_id)
            hard_labels = sample['hard_labels']
            avg_align_scores = sample['avg_align_scores']
            model_output_text = normalize_text(sample['model_output_text'])
            # Use a cursor to track the position of tokens in the text
            cursor = 0
            #avg_align_scores_sorted = sorted(sample['avg_align_scores'], key=lambda x: x['org_token_idx'])

            #for _ in avg_align_scores:
                #print(_['org_token'])

            
            for token_info in avg_align_scores:
                token_text = token_info['org_token'].strip("Ġ▁")
                token_score = token_info['avg_score']

                match = re.search(re.escape(token_text), model_output_text) if cursor == 0 else re.search(re.escape(token_text), model_output_text[cursor-1:])
                if match:
                    token_start = cursor + match.start()
                    token_end = cursor + match.end()

                    # Check if the token is hallucinated
                    if is_hallucinated(token_start, token_end, hard_labels):
                        model_data[model_id]["hallucinated"].append(token_score)
                    else:
                        model_data[model_id]["non_hallucinated"].append(token_score)

                    # Move the cursor forward
                    cursor = token_end
                else:
                    #raise ValueError(f"Token '{token_text}' not found in the text: {model_output_text}")
                    continue

                token_end = token_start + len(token_text)

                # Check if the token is hallucinated
                if is_hallucinated(token_start, token_end, hard_labels):
                    model_data[model_id]["hallucinated"].append(token_score)
                else:
                    model_data[model_id]["non_hallucinated"].append(token_score)

                # Move the cursor forward
                cursor = token_end
    
    if args.full_val:
        with open(os.path.join(sim_avgs.replace('test', 'val'), filename), 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                model_id = sample['model_id']
                print(model_id)
                hard_labels = sample['hard_labels']
                avg_align_scores = sample['avg_align_scores']
                model_output_text = normalize_text(sample['model_output_text'])
                # Use a cursor to track the position of tokens in the text
                cursor = 0
                #avg_align_scores_sorted = sorted(sample['avg_align_scores'], key=lambda x: x['org_token_idx'])

                #for _ in avg_align_scores:
                    #print(_['org_token'])

                
                for token_info in avg_align_scores:
                    token_text = token_info['org_token'].strip("Ġ▁")
                    token_score = token_info['avg_score']

                    match = re.search(re.escape(token_text), model_output_text) if cursor == 0 else re.search(re.escape(token_text), model_output_text[cursor-1:])
                    if match:
                        token_start = cursor + match.start()
                        token_end = cursor + match.end()

                        # Check if the token is hallucinated
                        if is_hallucinated(token_start, token_end, hard_labels):
                            model_data[model_id]["hallucinated"].append(token_score)
                        else:
                            model_data[model_id]["non_hallucinated"].append(token_score)

                        # Move the cursor forward
                        cursor = token_end
                    else:
                        #raise ValueError(f"Token '{token_text}' not found in the text: {model_output_text}")
                        continue

                    token_end = token_start + len(token_text)

                # Check if the token is hallucinated
                if is_hallucinated(token_start, token_end, hard_labels):
                    model_data[model_id]["hallucinated"].append(token_score)
                else:
                    model_data[model_id]["non_hallucinated"].append(token_score)

                # Move the cursor forward
                cursor = token_end
    
    model_ids = {
        'de': ['malteos/bloom-6b4-clp-german-oasst-v0.1', 'TheBloke/SauerkrautLM-7B-v1-GGUF', 'occiglot/occiglot-7b-de-en-instruct'],
        'fr': ['croissantllm/CroissantLLMChat-v0.1', 'bofenghuang/vigogne-2-13b-chat', 'mistralai/Mistral-Nemo-Instruct-2407', 'occiglot/occiglot-7b-eu5-instruct', 'meta-llama/Meta-Llama-3.1-8B-Instruct'],
        'en': ['TheBloke/Mistral-7B-Instruct-v0.2-GGUF', 'tiiuae/falcon-7b-instruct', 'togethercomputer/Pythia-Chat-Base-7B'],
        'ar': ['openchat/openchat-3.5-0106-gemma', 'arcee-ai/Arcee-Spark','SeaLLMs/SeaLLM-7B-v2.5'],
        'zh': ['Qwen/Qwen1.5-14B-Chat', 'baichuan-inc/Baichuan2-13B-Chat', '01-ai/Yi-1.5-9B-Chat', 'internlm/internlm2-chat-7b'],
        'hi': ['nickmalhotra/ProjectIndus', 'sarvamai/OpenHathi-7B-Hi-v0.1-Base', 'meta-llama/Meta-Llama-3-8B-Instruct'],
        'it': ['sapienzanlp/modello-italia-9b', 'meta-llama/Meta-Llama-3.1-8B-Instruct', 'rstless-research/DanteLLM-7B-Instruct-Italian-v0.1', 'Qwen/Qwen2-7B-Instruct'],
        'es': ['Iker/Llama-3-Instruct-Neurona-8b-v2', 'meta-llama/Meta-Llama-3-8B-Instruct', 'Qwen/Qwen2-7B-Instruct'],
        'sv': ['AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct-gguf', 'LumiOpen/Poro-34B-chat', 'LumiOpen/Viking-33B'],
        'fi': ['LumiOpen/Poro-34B-chat', 'Finnish-NLP/llama-7b-finnish-instruct-v0.2']

    }

    if args.threshold_metric != 'iou':

        for model_id in model_ids[lang]:
            model1_mean_hall = np.mean(model_data[model_id]["hallucinated"])
            model1_mean_non_hall = np.mean(model_data[model_id]["non_hallucinated"])

            model1_std_hall = np.std(model_data[model_id]["hallucinated"])
            model1_std_non_hall = np.std(model_data[model_id]["non_hallucinated"])

            """model1_threshold = model1_mean_non_hall - model1_std_non_hall
            print(f"{model_id}: Hallucinated {args.avg_method}: {model1_mean_hall}, std: {model1_std_hall}")
            print(f"{model_id}: Non-Hallucinated {args.avg_method}: {model1_mean_non_hall}, std: {model1_std_non_hall}")
            print(f"Threshold: {model1_threshold}")"""
        
            # Example: Replace with your hallucinated/non-hallucinated scores
            hallucinated_scores = model_data[model_id]['hallucinated']  # Similarity scores for hallucinated tokens
            non_hallucinated_scores = model_data[model_id]['non_hallucinated']  # Similarity scores for non-hallucinated tokens

            # Combine scores and labels
            all_scores = np.array(hallucinated_scores + non_hallucinated_scores)
            all_labels = np.array([1] * len(hallucinated_scores) + [0] * len(non_hallucinated_scores))

            # Sort thresholds from unique scores
            thresholds = thresholds = np.sort(np.unique(all_scores))

            # Initialize lists to store metrics
            precision_list, recall_list, f1_list = [], [], []

            for threshold in thresholds:
                # Predict based on threshold
                predictions = all_scores < threshold
                
                # Calculate True Positives, False Positives, False Negatives
                tp = np.sum((predictions == 1) & (all_labels == 1))
                fp = np.sum((predictions == 1) & (all_labels == 0))
                fn = np.sum((predictions == 0) & (all_labels == 1))
                
                # Precision, Recall
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)

            # Find the optimal threshold (maximize F1 Score)
            optimal_idx = np.argmax(f1_list)
            optimal_threshold = thresholds[optimal_idx]
            optimal_precision = precision_list[optimal_idx]
            optimal_recall = recall_list[optimal_idx]

            print(f"Model: {model_id}")
            print(f"Optimal Threshold: {optimal_threshold}")
            print(f"Precision: {optimal_precision}, Recall: {optimal_recall}, F1 Score: {f1_list[optimal_idx]}")
            print()
            if not args.fewer_altres:
                method = "_".join(sim_avgs.split('/')[-1].split('_')[-2:]) if not 'gpt' in args.sim_avg_path else 'gpt4o_' + "_".join(sim_avgs.split('/')[-1].split('_')[-2:])
            else:
                if not 'all' in sim_avgs:
                    method = "_".join(sim_avgs.split('/')[-1].split('_')[-4:]) if not 'gpt' in args.sim_avg_path else 'gpt4o_' + "_".join(sim_avgs.split('/')[-1].split('_')[-4:])
                else:
                    method = "_".join(sim_avgs.split('/')[-1].split('_')[-3:]) if not 'gpt' in args.sim_avg_path else 'gpt4o_' + "_".join(sim_avgs.split('/')[-1].split('_')[-3:])

            # Update JSON dictionary
            method = method + "_median" if args.avg_method == 'median' else method + "_mean"
            if args.full_val:
                method = method + '_full_val'
            print('METHOD'+f' {method}')
            if f'{lang}_{method}' not in sim_thresholds:
                sim_thresholds[f'{lang}_{method}'] = {}
            sim_thresholds[f'{lang}_{method}'][model_id] = float(optimal_threshold)
    
    else:
        for model_id in model_ids[lang]:
            # Convert to arrays if they aren't already
            hallucinated_scores = np.array(model_data[model_id]['hallucinated'])
            non_hallucinated_scores = np.array(model_data[model_id]['non_hallucinated'])
            
            # Combine scores and create labels (1 for hallucinated, 0 for non-hallucinated)
            all_scores = np.concatenate([hallucinated_scores, non_hallucinated_scores])
            all_labels = np.array([1]*len(hallucinated_scores) + [0]*len(non_hallucinated_scores))
            
            # Generate a sorted list of unique score values as candidate thresholds
            # Suppose these are your min and max values (floats).
            min_val, max_val = np.min(all_scores), np.max(all_scores)

            # 1) Using np.arange (caution: can create huge arrays if step is very small!)
            step = 0.000001 # pick a suitable step size
            thresholds = np.arange(min_val, max_val + step, step)
            
            best_iou = -1.0
            best_threshold = None
            
            # Evaluate each threshold
            for threshold in thresholds:
                # Predicted label: 1 (hallucinated) if score < threshold, else 0 (non-hallucinated)
                predictions = (all_scores < threshold)
                
                # Compute confusion matrix components
                tp = np.sum((predictions == 1) & (all_labels == 1))
                fp = np.sum((predictions == 1) & (all_labels == 0))
                fn = np.sum((predictions == 0) & (all_labels == 1))
                
                # Calculate IoU = TP / (TP + FP + FN)
                denom = (tp + fp + fn)
                iou = tp / denom if denom > 0 else 0.0
                
                # Track the best IoU and threshold
                if iou > best_iou:
                    best_iou = iou
                    best_threshold = threshold
        
            print(f"Model: {model_id}")
            print(f"Optimal Threshold: {best_threshold}")
            print(f"IoU: {best_iou}")
            print()

    
    with open(thresholds_file, "w", encoding="utf-8") as json_file:
        json.dump(sim_thresholds, json_file, indent=4, ensure_ascii=False)


    # Generate plots
    """output_pdf = f"{lang}_similarity_comparison.pdf"
    with PdfPages(output_pdf) as pdf:
        for model_id, scores in model_data.items():
            plt.figure()
            plt.boxplot(
                [scores["hallucinated"], scores["non_hallucinated"]],
                labels=["Hallucinated", "Non-Hallucinated"]
            )
            plt.title(f"Similarity Scores for {model_id}")
            plt.ylabel("Similarity Score")
            pdf.savefig()
            plt.close()

    print(f"Plots saved to {output_pdf}")"""