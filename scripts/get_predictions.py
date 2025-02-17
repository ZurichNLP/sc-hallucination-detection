import json
import os
import numpy as np
import itertools
from typing import List, Tuple
import pandas as pd
import argparse as ap
import stanza
import string
from itertools import groupby
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

from analyze_probs import log_transform

LANGS = ['ar', 'de', 'en', 'es', 'fi', 'fr', 'hi', 'it', 'sv', 'zh']

def split_validation_and_test_set():
    os.makedirs("./data/v2_splits/test", exist_ok=True)
    os.makedirs("./data/v2_splits/val", exist_ok=True)
    for filename in os.listdir("./data/val_setv2/"): 
        print('\n'+filename+'\n')
        path_to_file = os.path.join("./data/val_setv2/", filename)
        df = pd.read_json(path_to_file, lines=True, encoding='utf-8')
        # unique_model_names = df['model_id'].value_counts()

        sample_fraction = 0.5
        df_val = df.groupby('model_id').apply(lambda x: x.sample(frac=sample_fraction, random_state=42)).reset_index(drop=True)
        df_test = df[~df['id'].isin(df_val['id'])]
        
        df_val.to_json(f"./data/v2_splits/val/{filename}", orient='records', lines=True, force_ascii=False)
        df_test.to_json(f"./data/v2_splits/test/{filename}", orient='records', lines=True, force_ascii=False)

        print(f"\nProcessed {filename}: {len(df_val)} test rows, {len(df_test)} test rows. \
              \nEach model is representet in equal proportions in the vla set as in the test set.")


def split_validation_and_test_set_with_probs():
    """creates validation and test splits with the calculated probs"""
    # TODO
    pass


def fix_encoding_errors(token):
    # This function attempts to decode tokens that were double-encoded in UTF-8
    try:
        # Encode the string into bytes assuming it was wrongly interpreted as UTF-8
        encoded_bytes = token.encode('latin1')
        # Decode back into UTF-8 correctly
        return encoded_bytes.decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        # If decoding fails, return the token unchanged
        pass
    if 'Ã' in token:
    # Replace common double-encoding artifacts
        fixed_token = (token
                        .replace('Ã¼', 'ü')
                        .replace('Ã¶', 'ö')
                        .replace('Ã¤', 'ä')
                        .replace('ÃŸ', 'ß')
                        .replace('ÃŁ', 'ß')  # This could be a mis-encoding of 'ß'
                        .replace('Ã¡', 'á')
                        .replace('Ã©', 'é')
                        .replace('Ã±', 'ñ')
                        .replace('Ãģ', 'Á'))
        return fixed_token
    return token

def get_hard_labels(text_labels: List[Tuple[str, int]], original_model_output: str, granularity: str,) -> List[List[int]]:
    """
    Processes token or word probabilities to get the hard labels needed for the shared task.
    The output corresponds to hard labels in the shared task, so a list of integers denoting the start character position and end character position.
    Args:
        text_probs (List[Tuple[str, int]]): A list of tuples where each tuple contains a string token and its corresponding probability.
        token bool: Bool, which indicates whether the probabilities are on token or on word level.
    Returns:
        List[List[int, int]]: A list of lists where each inner list contains start and end positions for tokens of interest.
    """
    hard_labels = []
    current_pos = 0 

    for token_label in text_labels:
        token, label = token_label
        token = token.strip('▁ĠĊ').replace('<0x0A>', '\n')

        if granularity == 'token':
            token = token.strip('▁Ġ')

        if label == 0:
            current_pos += len(token)  
            continue

        start_idx = original_model_output.find(token, current_pos)

        fixed_token = ''
        if start_idx == -1:
            fixed_token = fix_encoding_errors(token)
            start_idx = original_model_output.find(fixed_token, current_pos)
            if start_idx == -1:
                print(original_model_output)
                print(original_model_output[current_pos:])
                # token not found; this could be a tokenization mismatch
                print(f"Warning: Token '{token}' (fixed: '{fixed_token}') not found in original text starting from position {current_pos}")
                continue

        end_idx = start_idx + len(token) if not fixed_token else start_idx + len(fixed_token) # End position is start + length of token

        hard_labels.append([start_idx, end_idx])  # note that start_idx would be the original version

        current_pos = end_idx

    return hard_labels

def get_hard_labels_sc(text_labels: List[Tuple[str, int]], original_model_output: str, granularity: str) -> List[List[int]]:
    """
    Processes token or word probabilities to get the hard labels needed for the shared task.
    The output corresponds to hard labels in the shared task, so a list of integers denoting the start character position and end character position.
    
    Args:
        text_labels (List[Tuple[str, int]]): A list of tuples where each tuple contains a string token and its corresponding label (0 or 1).
        original_model_output (str): The full original text.
        granularity (str): Specifies whether the labels are on the 'token' or 'word' level.
    
    Returns:
        List[List[int]]: A list of lists where each inner list contains the start and end positions for all parts of the text labeled as 1.
    """
    hard_labels = []
    current_pos = 0  # Keeps track of the current position in the original text

    for token_label in text_labels:
        token, label = token_label
        token = token.strip('▁ĠĊ').replace('<0x0A>', '\n')

        if granularity == 'token':
            token = token.strip('▁Ġ')

        # Find the start index of the token in the original text
        start_idx = original_model_output.find(token, current_pos)
        
        fixed_token = ''
        if start_idx == -1:
            fixed_token = fix_encoding_errors(token)
            start_idx = original_model_output.find(fixed_token, current_pos)
            if start_idx == -1:
                # Token not found; print a warning and skip to the next token
                print(f"Warning: Token '{token}' (fixed: '{fixed_token}') not found in original text starting from position {current_pos}")
                continue

        # Determine the end index of the token
        end_idx = start_idx + len(token) if not fixed_token else start_idx + len(fixed_token)

        # Add or merge the labeled token's start and end positions if label == 1
        if label == 1:
            if hard_labels and hard_labels[-1][1] >= start_idx:  # Overlapping or contiguous
                hard_labels[-1][1] = max(hard_labels[-1][1], end_idx)  # Extend the last span
            else:
                hard_labels.append([start_idx, end_idx])  # Add a new span

        # Update the current position to the end of the token
        current_pos = end_idx

    # Include the remaining text as labeled '1'
    if current_pos < len(original_model_output):
        remaining_text_start = current_pos
        remaining_text_end = len(original_model_output)
        if hard_labels and hard_labels[-1][1] >= remaining_text_start:  # Merge with the last span if overlapping
            hard_labels[-1][1] = max(hard_labels[-1][1], remaining_text_end)
        else:
            hard_labels.append([remaining_text_start, remaining_text_end])

    return hard_labels

def normalize_chinese_punctuation(text: str) -> str:
    replacements = {
        "（": "(",
        "）": ")",
        "，": ",",
        "。": ".",
        "：": ":",
        "；": ";",
        "？": "?",
        "！": "!",
        "【": "[",
        "】": "]",
        "～": "~",
        "“": "\"",
        "”": "\"",
        "‘": "'",
        "’": "'",
        "…": "...",
        "—": "-",   # or "--", depending on preference
        "一": "-",
        "——": "-",
        "｜": "|",   # Vertical bar
        "．": ".",   # Full stop
        "－": "-",   # Hyphen
        "·": ".",    # Middle dot to full stop
        "【分析】": "[Analysis]",  # Example of translating specific phrase
        "11．": "11.",  # Fix numbered list
        # Full-width numbers
        "１": "1", "２": "2", "３": "3", "４": "4", "５": "5",
        "６": "6", "７": "7", "８": "8", "９": "9", "０": "0",
        
        # Full-width symbols
        "．": ".", "，": ",", "：": ":", "；": ";", "！": "!", 
        "？": "?", "＋": "+", "－": "-", "＝": "=", "／": "/",
        "＼": "\\", "＠": "@", "＃": "#", "＄": "$", "％": "%",
        "＆": "&", "＊": "*", "（": "(", "）": ")", "［": "[",
        "］": "]", "｛": "{", "｝": "}", "｜": "|", "～": "~",
        "‘": "'", "’": "'", "“": "\"", "”": "\"", "…": "...",
        "——": "-", "【": "[", "】": "]", "《": "<", "》": ">",
        "·": ".",
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text
    
    for cjk_char, ascii_char in replacements.items():
        text = text.replace(cjk_char, ascii_char)
    
    return text

def get_hard_labels_sc(text_labels: List[Tuple[str, int]], original_model_output: str, granularity: str) -> List[List[int]]:
    """
    Processes token or word probabilities to get the hard labels needed for the shared task.
    The output corresponds to hard labels in the shared task, so a list of integers denoting the start character position and end character position.
    
    Args:
        text_labels (List[Tuple[str, int]]): A list of tuples where each tuple contains a string token and its corresponding label (0 or 1).
        original_model_output (str): The full original text.
        granularity (str): Specifies whether the labels are on the 'token' or 'word' level.
    
    Returns:
        List[List[int]]: A list of lists where each inner list contains the start and end positions for all parts of the text labeled as 1.
    """
    hard_labels = []
    current_pos = 0  # Keeps track of the current position in the original text

    for token_label in text_labels:
        token, label = token_label
        token = token.strip('▁ĠĊ').replace('<0x0A>', '\n')

        if granularity == 'token':
            token = token.strip('▁Ġ')

        # Find the start index of the token in the original text
        if lang == 'zh':
            token = normalize_chinese_punctuation(token)
            original_model_output = normalize_chinese_punctuation(original_model_output)
        original_model_output = original_model_output
        start_idx = original_model_output.find(token, current_pos)
        
        fixed_token = ''
        if start_idx == -1:
            fixed_token = fix_encoding_errors(token)
            start_idx = original_model_output.find(fixed_token, current_pos)
            if start_idx == -1:
                # Token not found; print a warning and skip to the next token
                print(f"Warning: Token '{token, len(token)}' (fixed: '{fixed_token, len(fixed_token)}') not found in original text starting from position {current_pos}")
                continue

        # Determine the end index of the token
        end_idx = start_idx + len(token) if not fixed_token else start_idx + len(fixed_token)

        # Add or merge the labeled token's start and end positions if label == 1
        if label == 1:
            if hard_labels and hard_labels[-1][1] >= start_idx:  # Overlapping or contiguous
                hard_labels[-1][1] = max(hard_labels[-1][1], end_idx)  # Extend the last span
            else:
                hard_labels.append([start_idx, end_idx])  # Add a new span

        # Update the current position to the end of the token
        current_pos = end_idx

    """# Include the remaining text as labeled '1'
    if current_pos < len(original_model_output):
        remaining_text_start = current_pos
        remaining_text_end = len(original_model_output)
        if hard_labels and hard_labels[-1][1] >= remaining_text_start:  # Merge with the last span if overlapping
            hard_labels[-1][1] = max(hard_labels[-1][1], remaining_text_end)
        else:
            hard_labels.append([remaining_text_start, remaining_text_end])"""

    return hard_labels


def get_non_labeled_spans(
    text_labels: List[Tuple[str, int]], 
    original_model_output: str, 
    granularity: str
) -> List[List[int]]:
    """
    Identifies spans in the original text that are not explicitly covered by the given text labels.
    
    Args:
        text_labels (List[Tuple[str, int]]): A list of tuples where each tuple contains a string token and its corresponding label (0 or 1).
        original_model_output (str): The full original text.
        granularity (str): Specifies whether the labels are on the 'token' or 'word' level.
    
    Returns:
        List[List[int]]: A list of lists where each inner list contains the start and end positions for spans not covered by the labels.
    """
    non_labeled_spans = []
    hard_labels = []
    current_pos = 0  # Keeps track of the current position in the original text

    for token_label in text_labels:
        token, label = token_label
        token = token.strip('▁ĠĊ').replace('<0x0A>', '\n').strip()

        if granularity == 'token':
            token = token.strip('▁Ġ')

        # Find the start index of the token in the original text
        start_idx = original_model_output.find(token, current_pos)
        
        fixed_token = ''
        if start_idx == -1:
            fixed_token = fix_encoding_errors(token)
            start_idx = original_model_output.find(fixed_token, current_pos)
            if start_idx == -1:
                # Token not found; print a warning and skip to the next token
                print(f"Warning: Token '{token}' (fixed: '{fixed_token}') not found in original text starting from position {current_pos}")
                continue

        # Determine the end index of the token
        end_idx = start_idx + len(token) if not fixed_token else start_idx + len(fixed_token)

        # Add the labeled token's start and end positions only if label == 1
        if label == 1:
            hard_labels.append([start_idx, end_idx])

        # Update the current position to the end of the token
        current_pos = end_idx

    # Include the remaining text as labeled '1'
    if current_pos < len(original_model_output):
        hard_labels.append([current_pos, len(original_model_output)])

    # Compute non-labeled spans
    last_end = 0
    for start, end in hard_labels:
        if last_end < start:  # There's a gap between the last end and the current start
            non_labeled_spans.append([last_end, start])
        last_end = end

    # Add any trailing text at the end of the document
    if last_end < len(original_model_output):
        non_labeled_spans.append([last_end, len(original_model_output)])

    return non_labeled_spans



def get_labels_based_on_prob_minima(text_probs: List[Tuple[str, float]], threshold: float) -> List[Tuple[str, int]]:
    """
    Labels each token with either 0 or 1 based on its probability.
    
    Tokens with a raw probability greater than the threshold (relative to the highest probability in the sentence) are labeled with 0.
    Tokens with a raw probability lower than the threshold are labeled with 1.
    
    Args:
        text_probs (List[Tuple[str, float]]): A list of tuples where each tuple contains a token and its corresponding probability.
        threshold (float): The threshold value (in percentage) used to classify tokens. Should be between 0 and 100.
    
    Returns:
        List[Tuple[str, int, int]]: A list of tuples where each tuple contains a token and its corresponding label (0 or 1) and its raw probability.
    """
    log_probs = log_transform([prob for _, prob in text_probs])
    pos_log_probs = [-prob for prob in log_probs]
    max_prob = max(pos_log_probs)
    
    threshold = 1 - threshold
    threshold_value = threshold * max_prob
    
    labeled_tokens = [(token_prob[0], 1 if log_prob >= threshold_value else 0) for log_prob, token_prob in zip(pos_log_probs, text_probs)]    
    
    return labeled_tokens


def get_prob_difference(combination1, combination2):
    # TODO (also not yet sure about function arguments)
    pass


def fuse_tokens_to_words(token_prob_list, approach):
    word_probs = []
    current_word = ""
    current_probs = []

    for token, prob in token_prob_list:
        if token.startswith("▁") or token.startswith("<") or token.startswith("\u010a") or token.startswith("\u0120") or token.startswith("Ċ") or token.startswith(" ") or not token.startswith("##"):
            if current_word:
                if approach == 'avg':
                    word_probs.append((current_word, sum(current_probs) /len(current_probs)))
                elif approach == 'min_binary':
                    word_probs.append((current_word, 1 if sum(current_probs) > 0 else 0))
                elif approach == 'min':
                    word_probs.append((current_word, min(current_probs)))
            current_word = token[1:] if token.startswith("▁") or token.startswith("\u0120") else token 
            current_probs = [prob]
        else:
            current_word += token.strip("##")
            current_probs.append(prob)
    
    if current_word:
        if approach == 'avg':
            word_probs.append((current_word, sum(current_probs) /len(current_probs)))
        elif approach == 'min_binary':
            word_probs.append((current_word, 1 if sum(current_probs) > 0 else 0))
        elif approach == 'min':
            word_probs.append((current_word, min(current_probs)))
        print('word', current_word)
    
    return word_probs


def fuse_words_to_phrases(model_output_text, word_prob_list, lang, nlp):
    # Step 1: Annotate the text using Stanza and retrieve words with their phrase labels
    doc = nlp(model_output_text)
    
    # Extract leaf words and their phrase labels
    word_phrase_labels = []
    
    def extract_leaf_labels(tree, phrase_label):
        if tree.is_leaf():
            word_phrase_labels.append((str(tree), phrase_label))  # Append word and phrase label
        else:
            for child in tree.children:
                extract_leaf_labels(child, tree.label)  # Recursively process children
    
    for sentence in doc.sentences:
        extract_leaf_labels(sentence.constituency, "ROOT")
    
    # Step 2: Match words with `word_prob_list` to create triples
    result = []
    for word, phrase_label in word_phrase_labels:
        for word_prob in word_prob_list:
            if word in word_prob[0]:  # Match based on substring
                result.append((word_prob[0], word_prob[1], phrase_label))
                break  # Stop once a match is found
    
    # Step 3: Adjust labels for contiguous elements with the same phrase label
    adjusted_result = []
    for phrase_label, group in groupby(result, key=lambda x: x[2]):  # Group by phrase label
        group = list(group)
        if any(item[1] == 1 for item in group):  # If any label in the group is 1
            adjusted_group = [(item[0], 1) for item in group]
        else:  # If all labels in the group are 0
            adjusted_group = [(item[0], 0) for item in group]
        adjusted_result.extend(adjusted_group)
    
    return adjusted_result


def get_probabilities_from_logits(logits):
    """Softmax but with numerical stability (subtracting the max value) from the logits"""
    pass
    # TODO


def get_iqr_outliers(values, model_output_tokens, percentile_l=25, percentile_u=75, multiplier=1.5, hl='low'):
    """Find the lower bound outliers using IQR."""

    assert len(values) == len(model_output_tokens)

    Q1_logits = np.percentile(values, percentile_l)
    Q3_logits = np.percentile(values, percentile_u)
    IQR_logits = Q3_logits - Q1_logits
    lower_bound_logits = Q1_logits - multiplier * IQR_logits
    upper_bound_logits = Q3_logits + multiplier * IQR_logits
    

    if hl == 'low':
        iqr_outlier_indices = np.where(values < lower_bound_logits)[0]
        labeled_model_output_text = [(token, 1 if index in iqr_outlier_indices else 0) for index, token in enumerate(model_output_tokens)]
    elif hl == 'high':
        iqr_outlier_indices = np.where(values > upper_bound_logits)[0]
        labeled_model_output_text = [(token, 0 if index in iqr_outlier_indices else 1) for index, token in enumerate(model_output_tokens)]
    elif hl == 'high_diff':
        iqr_outlier_indices = np.where(values > upper_bound_logits)[0]
        labeled_model_output_text = [(token, 1 if index in iqr_outlier_indices else 0) for index, token in enumerate(model_output_tokens)]
   
    assert len(values) == len(model_output_tokens) == len(labeled_model_output_text)
    #assert sum(label for token, label in labeled_model_output_text) == len(iqr_outlier_indices) if hl == 'low' else sum(1 for token, label in labeled_model_output_text if label == 0) == len(iqr_outlier_indices)

    return labeled_model_output_text


def get_logits_below_threshold(model_id, lang, logits, tokens):
    id = model_id.replace('/', "_") + "_" + lang.upper()
    threshold = {
    "togethercomputer_Pythia-Chat-Base-7B_EN": 2.0124,
    "tiiuae_falcon-7b-instruct_EN": -10.4003,
    "TheBloke_Mistral-7B-Instruct-v0.2-GGUF_EN": 0.9790,
    "meta-llama_Meta-Llama-3-8B-Instruct_ES": 21.1932,
    "Qwen_Qwen2-7B-Instruct_ES": 18.1121,
    "Iker_Llama-3-Instruct-Neurona-8b-v2_ES": 21.7673,
    "mistralai_Mistral-Nemo-Instruct-2407_FR": 19.7600,
    "bofenghuang_vigogne-2-13b-chat_FR": 17.4817,
    "occiglot_occiglot-7b-eu5-instruct_FR": 21.7343,
    "croissantllm_CroissantLLMChat-v0.1_FR": 60.4928,
    "meta-llama_Meta-Llama-3.1-8B-Instruct_FR": 19.4448,
    "Qwen_Qwen1.5-14B-Chat_ZH": 18.9561,
    "THUDM_chatglm3-6b_ZH": 14.5547,
    "01-ai_Yi-1.5-9B-Chat_ZH": 15.9635,
    "baichuan-inc_Baichuan2-13B-Chat_ZH": 21.1992,
    "internlm_internlm2-chat-7b_ZH": 286.9067,
    }
    labeled_tokens = [(token, 1 if logit < threshold[id] else 0) for logit, token in zip(logits, tokens)]
    return labeled_tokens


# hypotheses
# 1. the lowest token probability of each original input output combination under 
# a. the original model
# b. the other two models

# 4. the largest difference of token/word probabilities between original combination and fitting gpt4o input

# 5. the lowest probabilities if the probabilities of input and output combinations are averaged ((maybe weighted avg))
# a. original + no prompt
# b. original + gpt4o
# c. original + gpt4o + no prompt
# d. gpt4o + no prompt
# e. do this for each model

# scrambled output ist nicht richtig motiviert
# aber umformulierungen wären gut:
    # word alignment könnte man benutzen um es zurück zu mappen
    # es gibt verschiedene formate für word alignment wie sphynx und dann gpt fragen
    # bert oder roberta embeddings benutzen mit SimAlign


def post_processing(hard_labels):
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


def fuse_align_tokens(labeled_tokens, fusing_strategy):
    # Initialize variables
    fused_words = []
    current_word = ""
    current_labels = []

    for token, label in labeled_tokens:
        # Check if token starts a new word (▁ indicates the start of a word in BPE)
        if token.startswith("▁"):
            # Fuse the previous word and calculate its label
            if current_word:
                if fusing_strategy == 'majority':
                    majority_label = 1 if sum(current_labels) >= len(current_labels) / 2 else 0
                    fused_words.append((current_word, majority_label))
                elif fusing_strategy == 'min':
                    majority_label = 1 if any(current_labels) else 0
                    fused_words.append((current_word, majority_label))
            # Start a new word
            current_word = token[1:]  # Remove the ▁ from the start
            current_labels = [label]
        else:
            # Continue building the current word
            current_word += token
            current_labels.append(label)

    # Handle the last word
    if current_word:
        if fusing_strategy == 'majority':
            majority_label = 1 if sum(current_labels) >= len(current_labels) / 2 else 0
            fused_words.append((current_word, majority_label))
        elif fusing_strategy == 'min':
            majority_label = 1 if any(current_labels) else 0
            fused_words.append((current_word, majority_label))

    return fused_words

def fuse_align_tokens_avg(threshold, avg_sim_scores_per_token, fusing_strategy):
    # variable names are a little off here
    fused_words = []
    current_word = ""
    current_labels = []
    for _ in avg_sim_scores_per_token:
        token = _['org_token']
        label = _['avg_score']
        # Check if token starts a new word (▁ indicates the start of a word in BPE)
        if token.startswith("▁"):
            # Fuse the previous word and calculate its label
            if current_word:
                if fusing_strategy == 'mean':
                    majority_label = 1 if np.mean(current_labels) < threshold else 0
                    fused_words.append((current_word, majority_label))
                elif fusing_strategy == 'median':
                    majority_label = 1 if np.median(current_labels) < threshold else 0
                    fused_words.append((current_word, majority_label))
            # Start a new word
            current_word = token[1:]  # Remove the ▁ from the start
            current_labels = [label]
        else:
            # Continue building the current word
            current_word += token
            current_labels.append(label)

    # Handle the last word
    if current_word:
        if fusing_strategy == 'mean':
            majority_label = 1 if np.mean(current_labels) < threshold else 0
            fused_words.append((current_word, majority_label))
        elif fusing_strategy == 'median':
            majority_label = 1 if np.median(current_labels) < threshold else 0
            fused_words.append((current_word, majority_label))

    return fused_words

def handle_unk_model(model_id, lang):
    model_id = model_id.strip()
    if lang == 'fa':
        print(model_id)
        if 'PersianMind-v1.0' in model_id: 
            return 'PersianMind-v1.0', 'ar'
        elif 'Meta-Llama-3.1-8B-Instruct' in model_id:
            return 'meta-llama/Meta-Llama-3.1-8B-Instruct', 'hi'
        elif 'Llama-3.2-3B-Instruct' in model_id:
            return 'PersianMind-v1.0', 'ar'
        elif 'aya-23-35B' in model_id:
            return 'PersianMind-v1.0', 'ar'
        elif 'aya-23-8B' in model_id:
            return 'PersianMind-v1.0', 'ar'
        elif 'Qwen2.5-7B-Instruct' in model_id:
            return 'Qwen/Qwen2-7B-Instruct', 'it'
        
    elif lang == 'ca':
        if 'Meta-Llama-3-8B-Instruct' in model_id: 
            return 'meta-llama/Meta-Llama-3-8B-Instruct', 'es'
        elif 'occiglot-7b-es-en-instruct' in model_id:
            return 'occiglot/occiglot-7b-eu5-instruct', 'fr'
        elif 'Mistral-7B-Instruct-v0.3' in model_id:
            return 'mistralai/Mistral-7B-Instruct-v0.3', 'en'

    elif lang == 'eu':
        if 'Meta-Llama-3-8B-Instruct' in model_id: 
            return 'meta-llama/Meta-Llama-3-8B-Instruct', 'es'
        elif 'google/gemma-7b-it' in model_id:
            print(model_id)
            return 'google/gemma-7b-it', 'es'
        
    elif lang == 'cs':
        if 'Meta-Llama-3-8B-Instruct' in model_id: 
            return 'meta-llama/Meta-Llama-3-8B-Instruct', 'es'
        elif 'Mistral-7B-Instruct-v0.3' in model_id:
            return 'mistralai/Mistral-7B-Instruct-v0.3', 'en'

    return model_id, lang




if __name__ == "__main__":
    p = ap.ArgumentParser()
    p.add_argument('strategy',type=str, help="Hallucination detections strategy (e.g.: 'full-seq-as-h', 'prob-minima', 'no-prompt-prob-diff', 'gpt-prompt-prob-diff').")
    p.add_argument('split', type=str, help="Choose either 'val' or 'test'.", default='val')
    p.add_argument('--langs', nargs='+', help='Choose either one language abbreviation (e.g.: "de")or a selection separated by commas (without whitespace) (e.g.:"en,de,fr"). If argument is not used, all languages will be considered.')
    p.add_argument('--sim-scores', action='store_true', help='If set, the similarity scores will be the soft labels. Works only for alignment based approaches.')
    p.add_argument('--fusing-strategy', type=str, default='majority', help='Choose between majority, min, median and mean. With majority the word label will be based on the majority token-level labels in the word, with min, the word will be labelled 1 if it contains one token that is labelled 1. Mean and median will calculate the mean and median sim score avgs of the word and compare it to the threshold. Only works, if the labelling is word-based.')
    p.add_argument('--same-simavgs', action='store_true', help='If set the sim score averages will be based on fewer samples.')

    a = p.parse_args()
    LANGS = LANGS if not a.langs else a.langs
    # split_validation_and_test_set()
    for lang in LANGS:
        outdir = f'./preds_v2/{a.split}/{a.strategy}' if not a.sim_scores else f'./preds_v2/{a.split}/{a.strategy}-soft_sim' 
        os.makedirs(outdir, exist_ok=True)
        outfile = f'{outdir}/mushroom.{lang}-val.v2.jsonl'

        if 'prob' in a.strategy:
            if lang in ['sv']:
                continue
            with open(f'./data/v1_splits/{a.split}/mushroom.{lang}-val.v1.jsonl', 'r', encoding='utf-8') as h:
                samples_w_probs = {json.loads(line)['id']:json.loads(line) for line in h}

        file = f'./data/v2_splits/{a.split}/mushroom.{lang}-val.v2.jsonl' if a.split != 'test_jan25' else f'./data/test_jan25/mushroom.{lang}-tst.v1.jsonl'
        with open(file, 'r', encoding='utf-8') as f, \
            open(outfile, 'w', encoding='utf-8') as g:

            processors = 'tokenize,ner' if a.strategy == 'entities' else 'tokenize,pos,constituency'
            nlp = stanza.Pipeline(lang=lang, processors=processors) if a.strategy in ['logit_threshold_train_consts', 'entities'] and lang != 'hi' else None

            if lang == 'hi' and (a.strategy == 'entities' or a.strategy == 'entities_num'):
                tokenizer = AutoTokenizer.from_pretrained("cfilt/HiNER-original-xlm-roberta-large")
                model = AutoModelForTokenClassification.from_pretrained("cfilt/HiNER-original-xlm-roberta-large")
            for line in f:
                sample = json.loads(line)
                pred = {k: sample[k] for k in itertools.islice(sample, 5)}

                # dummy test for the case if all chars are labelled as hallucinations
                if a.strategy == 'full-seq-as-h':
                    hard_labels = [[0, len(sample['model_output_text'])]]

                # 1. the lowest token using iqr
                # a) logits
                """labeled_text = get_iqr_outliers(sample['model_output_logits'], sample['model_output_tokens'],multiplier=0)
                hard_labels = get_hard_labels(labeled_text, sample['model_output_text'], 'token')"""
                # a) label highest logits as true
                """labeled_text = get_iqr_outliers(sample['model_output_logits'], sample['model_output_tokens'], multiplier=0, hl='high')
                hard_labels = get_hard_labels(labeled_text, sample['model_output_text'], 'token')"""
                
                if 'prob' in a.strategy:
                    if samples_w_probs[sample['id']]["model_id"] in ['mistral', 'LumiOpen/Poro-34B-chat', 'Qwen/Qwen1.5-14B-Chat', 'baichuan-inc/Baichuan2-13B-Chat']:
                        continue

                    orig_model_token_probs = samples_w_probs[sample['id']][f'{sample["model_id"]}_token_probs']

                # 2. the lowest word probability of each original input output combination under 
                    # a. the original model
                if a.strategy == 'prob-minima':
                    if samples_w_probs[sample['id']]["model_id"] in ['mistral', 'LumiOpen/Poro-34B-chat', 'Qwen/Qwen1.5-14B-Chat', 'baichuan-inc/Baichuan2-13B-Chat']:
                        continue
                    assert samples_w_probs[sample['id']]['model_output_text'] == sample['model_output_text']
                    #word_probs = fuse_tokens_to_words(orig_model_token_probs)
                    #word_labels = get_labels_based_on_prob_minima(word_probs, 0.95)
                    labeled_tokens = get_iqr_outliers([prob for token, prob in orig_model_token_probs], [token for token, prob in orig_model_token_probs], percentile_l=25, percentile_u=75, multiplier=0, hl='low')
                    labeled_words = fuse_tokens_to_words(labeled_tokens, approach='min_binary')
                    hard_labels = get_hard_labels(labeled_words, original_model_output=sample['model_output_text'], granularity='word')
                
                # 3. the lowest difference of token/word probabilities between original combination and no prompt
                if a.strategy == 'no-prompt-prob-diff':
                    orig_model_token_probs_no_prompt = samples_w_probs[sample['id']][f'{sample["model_id"]}_token_probs_noprompt']
                    while orig_model_token_probs_no_prompt and (orig_model_token_probs_no_prompt[0][0] == '<s>' or orig_model_token_probs_no_prompt[0][0] == '▁'):
                        orig_model_token_probs_no_prompt.pop(0)
                    orig_model_word_probs_no_prompt = fuse_tokens_to_words(orig_model_token_probs_no_prompt, approach='min')
                    orig_model_word_probs = fuse_tokens_to_words(orig_model_token_probs, approach='min')
                    word_prob_diffs = [(abs(prob-prob_np), token) for (token, prob), (token_np, prob_np) in zip(orig_model_word_probs, orig_model_word_probs_no_prompt)]
                    labeled_words = get_iqr_outliers([prob for prob, token in word_prob_diffs], [token for prob, token in word_prob_diffs], percentile_l=25, percentile_u=75, multiplier=0, hl='low')
                    hard_labels = get_hard_labels(labeled_words, original_model_output=sample['model_output_text'], granularity='word')
                
                # 4. the smallest difference of token/word probabilities between original combination and fitting gpt4o input
                if a.strategy == 'gpt-prompt-prob-diff': 
                    orig_model_token_probs_gpt4o = samples_w_probs[sample['id']][f'{sample["model_id"]}_token_probs_gpt4o']
                    assert len(orig_model_token_probs_gpt4o) == len(orig_model_token_probs)
                    token_prob_diffs = [(abs(prob-prob_gpt4o), token) for (token, prob), (token_gpt4o, prob_gpt4o) in zip(orig_model_token_probs, orig_model_token_probs_gpt4o)]
                    print(token_prob_diffs)
                    labeled_tokens = get_iqr_outliers([prob for prob, token in token_prob_diffs], [token for prob, token in token_prob_diffs], percentile_l=25, percentile_u=75, multiplier=0, hl='low')
                    print(labeled_tokens)
                    labeled_words = fuse_tokens_to_words(labeled_tokens, approach='min_binary')
                    hard_labels = get_hard_labels(labeled_words, original_model_output=sample['model_output_text'], granularity='word')

                # 5. logit threshold based on training data
                if a.strategy == 'logit_threshold_train':
                    if lang in ["en", "es", "fr", "zh"]:
                        labeled_tokens = get_logits_below_threshold(sample['model_id'], lang, sample['model_output_logits'], sample['model_output_tokens'])
                        labeled_words = fuse_tokens_to_words(labeled_tokens, approach='min_binary')
                        hard_labels = get_hard_labels(labeled_words, original_model_output=sample['model_output_text'], granularity='word')
                
                if a.strategy == 'logit_threshold_train_consts':
                    if lang in ["en","zh","es","fr"]:
                        labeled_tokens = get_logits_below_threshold(sample['model_id'], lang, sample['model_output_logits'], sample['model_output_tokens'])
                        labeled_words = fuse_tokens_to_words(labeled_tokens, approach='min_binary')
                        labeled_constituents = fuse_words_to_phrases(sample['model_output_text'], labeled_words, lang, nlp)
                        print(labeled_constituents)
                        hard_labels = get_hard_labels(labeled_constituents, original_model_output=sample['model_output_text'], granularity='word')

                # 6. Label all named entities and numbers as hallucinations
                if a.strategy == 'entities':
                    if lang in ['hi']:
                        tokens = tokenizer(sample['model_output_text'], return_tensors="pt", truncation=True)
                        with torch.no_grad():
                            logits = model(**tokens).logits
                        predicted_token_class_ids = logits.argmax(-1)
                        predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
                        actual_tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])  # Convert token IDs to actual tokens
                        labeled_tokens = [(token, 1 if label != 'O' else 0) for token, label in zip(actual_tokens, predicted_tokens_classes)]
                        labeled_words = fuse_tokens_to_words(labeled_tokens, approach='min_binary')[1:-1]
                        hard_labels = post_processing(get_hard_labels(labeled_words, original_model_output=sample['model_output_text'], granularity='word'))
                    else:
                        doc = nlp(sample['model_output_text'])
                        labeled_tokens = []
                        for sent in doc.sentences:
                            entity_token_ids = {token.id for ent in sent.ents for token in ent.tokens}
                            for token in sent.tokens:
                                if token.id in entity_token_ids:
                                    labeled_tokens.append((token.text, 1))  # Label named entity tokens as 1
                                elif token.words[0].upos == 'NUM':
                                    labeled_tokens.append((token.text, 1))
                                else:
                                    labeled_tokens.append((token.text, 0)) 
                            hard_labels = post_processing(get_hard_labels(labeled_tokens, original_model_output=sample['model_output_text'], granularity='word'))

                # 7. self-consistency threshold based approach
                if a.strategy.startswith('sc-threshold'):
                    thresholds = json.load(open('./scripts/sim_thresholds_by_model.json', 'r'))
                    if not 'median' in a.strategy:
                        avg_sim_file = f'./data/alt_res_{a.split}_setv2/sim_avgs_a/mushroom.{lang}-val.v2.jsonl' if a.strategy == 'sc-threshold' else f'./data/alt_res_{a.split}_setv2/sim_avgs_f/mushroom.{lang}-val.v2.jsonl'
                    elif ('all' in a.strategy or ('t0.' in a.strategy)) and not a.same_simavgs: 
                        avg_sim_file = f'./data/alt_res_{a.split}_setv2/sim_avgs_f_{a.strategy.replace("sc-threshold-f-", "").replace("-", "_").replace("_same_t", "")}/mushroom.{lang}-val.v2.jsonl' if not 'test_jan25' in a.split else f'./data/alt_res_{a.split}_setv2/sim_avgs_f_{a.strategy.replace("sc-threshold-f-", "").replace("-", "_").replace("_same_t", "")}/mushroom.{lang}-tst.v1.jsonl'
                    else:
                        if 'test_jan25' in a.split:
                            avg_sim_file = f'./data/alt_res_{a.split}_setv2/sim_avgs_f_median/mushroom.{lang}-tst.v1.jsonl'
                        else:
                            avg_sim_file = f'./data/alt_res_{a.split}_setv2/sim_avgs_f_median/mushroom.{lang}-val.v2.jsonl'
                    with open(avg_sim_file, 'r', encoding='utf-8') as sc:
                        for line in sc:
                            sample_sc = json.loads(line)
                            if sample_sc['id'] == sample['id']:
                                if a.strategy == 'sc-threshold':
                                    threshold_suffix = ''
                                elif a.strategy == 'sc-threshold-f':
                                    threshold_suffix = '_avgs_f'
                                elif a.strategy == 'sc-threshold-f-median' or 'same-t' in a.strategy:
                                    threshold_suffix = '_f_median'
                                elif a.strategy == 'sc-threshold-f-all-median':
                                    threshold_suffix = '_f_all_median'
                                elif a.strategy == 'sc-threshold-f-median-full-val':
                                    threshold_suffix = '_avgs_f_median_full_val'
                                elif 'p0.' in a.strategy and 't0.' in a.strategy:
                                    threshold_suffix = a.strategy.replace('sc-threshold-f', '').replace('k50_', '').replace('-', '_')
                                avg_sim_scores_per_token = sample_sc['avg_align_scores']
                                if lang in ['fa', 'ca', 'eu', 'cs']:
                                    sub_model_id, sub_lang = handle_unk_model(sample['model_id'], lang)
                                if a.fusing_strategy == 'mean' or a.fusing_strategy == 'median':
                                    labeled_words_from_alignment = fuse_align_tokens_avg(thresholds[f"{lang if not lang in ['fa', 'ca', 'eu', 'cs'] else sub_lang}{threshold_suffix}"][sample['model_id'] if not lang in ['fa', 'ca', 'eu', 'cs'] else sub_model_id], avg_sim_scores_per_token, a.fusing_strategy)
                                else:
                                    if lang in ['fa', 'ca', 'eu', 'cs']:
                                        sub_model_id, sub_lang = handle_unk_model(sample['model_id'], lang)
                                    if sample['model_id'] in thresholds[f"{lang if not lang in ['fa', 'ca', 'eu', 'cs'] else sub_lang}{threshold_suffix}"]:
                                        labeled_tokens_from_alignment = [(_['org_token'], 1) if _['avg_score']<thresholds[f"{lang if not lang in ['fa', 'ca', 'eu', 'cs'] else sub_lang}{threshold_suffix}"][sample['model_id']] else (_['org_token'], 0) for _ in avg_sim_scores_per_token]
                                    else:
                                        threshold_values = []
                                        for key, value in thresholds[f"{lang if not lang in ['fa', 'ca', 'eu', 'cs'] else sub_lang}{threshold_suffix}"].items():
                                            threshold_values.append(value)
                                        threshold_avg = sum(threshold_values)/len(threshold_values)
                                        labeled_tokens_from_alignment = [(_['org_token'], 1) if _['avg_score']<threshold_avg else (_['org_token'], 0) for _ in avg_sim_scores_per_token]
                                    labeled_words_from_alignment = fuse_align_tokens(labeled_tokens_from_alignment, a.fusing_strategy) # TODO: comment this in for word level labels and change the first argument in the next line to labeled_words_from_alignment
                                hard_labels = get_hard_labels_sc(labeled_words_from_alignment, original_model_output=sample['model_output_text'], granularity='token')
                                hard_labels = post_processing(hard_labels)
                                if a.sim_scores:
                                    sim_score_per_token = [(_['org_token'], _['avg_score']) for _ in avg_sim_scores_per_token]
                                    assert len(labeled_tokens_from_alignment) == len(sim_score_per_token)
                                    #sim_score_per_word = fuse_align_tokens(sim_score_per_token)
                                    #assert len(labeled_words_from_alignment) == len(sim_score_per_word)

                # 8. gpt-consistency threshold based approach
                if a.strategy.startswith('gpt-consistency'): 
                    print(a.strategy)
                    thresholds = json.load(open('./scripts/sim_thresholds_by_model.json', 'r'))
                    if not 'median' in a.strategy:
                        avg_sim_file = f'./data/alt_res_{a.split}_setv2/sim_avgs_gpt4o_{a.strategy[-1] if a.strategy.endswith("f") else "a"}/mushroom.{lang}-val.v2.jsonl' if not 'paras' in a.strategy else f'./data/alt_res_{a.split}_setv2/sim_avgs_gpt4o-paras_{a.strategy[-1] if a.strategy.endswith("f") else "a"}/mushroom.{lang}-val.v2.jsonl'
                    elif ('all' in a.strategy or 't0.' in a.strategy) and not a.same_simavgs: 
                        avg_sim_file = f'./data/alt_res_{a.split}_setv2/sim_avgs_gpt4o_f_{a.strategy.replace("gpt-consistency-f-", "").replace("-", "_").replace("_same_t", "")}/mushroom.{lang}-val.v2.jsonl'
                    else:
                        if 'test_jan25' in a.split:
                            avg_sim_file = f'./data/alt_res_{a.split}_setv2/sim_avgs_gpt4o_f_median/mushroom.{lang}-tst.v1.jsonl'
                        else:
                            avg_sim_file = f'./data/alt_res_{a.split}_setv2/sim_avgs_gpt4o_f_median/mushroom.{lang}-val.v2.jsonl' if not 'paras' in a.strategy else f'./data/alt_res_{a.split}_setv2/sim_avgs_gpt4o-paras_f_median/mushroom.{lang}-val.v2.jsonl'

                    with open(avg_sim_file, 'r', encoding='utf-8') as sc:
                        for line in sc:
                            sample_sc = json.loads(line)
                            if sample_sc['id'] == sample['id']:
                                if a.strategy == 'gpt-consistency':
                                    threshold_suffix = '_gpt'
                                elif a.strategy == 'gpt-consistency-paras':
                                    threshold_suffix = '_gpt4o-paras_a'
                                elif a.strategy == 'gpt-consistency-f':
                                    threshold_suffix = '_gpt_f'
                                elif a.strategy == 'gpt-consistency-paras-f':
                                    threshold_suffix = '_gpt4o-paras_f'
                                elif a.strategy == 'gpt-consistency-f-median' or 'same-t' in a.strategy:
                                    threshold_suffix = '_gpt4o_f_median'
                                elif a.strategy == 'gpt-consistency-f-all-median':
                                    threshold_suffix = '_gpt4o_f_all_median'
                                elif a.strategy == 'gpt-consistency-f-median-full-val':
                                    threshold_suffix = '_avgs_f_median_full_val'
                                elif 'p0.' in a.strategy and 't0.' in a.strategy:
                                    threshold_suffix = a.strategy.replace('gpt-consistency-f', '_gpt4o').replace('-', '_').replace('k50_', '')
                                avg_sim_scores_per_token = sample_sc['avg_align_scores']
                                if lang in ['fa', 'ca', 'eu', 'cs']:
                                    sub_model_id, sub_lang = handle_unk_model(sample['model_id'], lang)
                                if a.fusing_strategy == 'mean' or a.fusing_strategy == 'median':
                                    labeled_words_from_alignment = fuse_align_tokens_avg(thresholds[f"{lang if not lang in ['fa', 'ca', 'eu', 'cs'] else sub_lang}{threshold_suffix}"][sample['model_id'] if not lang in ['fa', 'ca', 'eu', 'cs'] else sub_model_id], avg_sim_scores_per_token, a.fusing_strategy)
                                else:
                                    if sample['model_id'] in thresholds[f"{lang if not lang in ['fa', 'ca', 'eu', 'cs'] else sub_lang}{threshold_suffix}"]:
                                        labeled_tokens_from_alignment = [(_['org_token'], 1) if _['avg_score']<thresholds[f"{lang if not lang in ['fa', 'ca', 'eu', 'cs'] else sub_lang}{threshold_suffix}"][sample['model_id']] else (_['org_token'], 0) for _ in avg_sim_scores_per_token]
                                    else:
                                        threshold_values = []
                                        for key, value in thresholds[f"{lang if not lang in ['fa', 'ca', 'eu', 'cs'] else sub_lang}{threshold_suffix}"].items():
                                            threshold_values.append(value)
                                        threshold_avg = sum(threshold_values)/len(threshold_values)
                                        labeled_tokens_from_alignment = [(_['org_token'], 1) if _['avg_score']<threshold_avg else (_['org_token'], 0) for _ in avg_sim_scores_per_token]
                                    labeled_words_from_alignment = fuse_align_tokens(labeled_tokens_from_alignment, a.fusing_strategy) # TODO: comment this in for word level labels and change the first argument in the next line to labeled_words_from_alignment
                                hard_labels = get_hard_labels_sc(labeled_words_from_alignment, original_model_output=sample['model_output_text'], granularity='token')
                                hard_labels = post_processing(hard_labels)
                                if a.sim_scores:
                                    # token level
                                    sim_score_per_token = [(_['org_token'], _['avg_score']) for _ in avg_sim_scores_per_token]
                                    assert len(labeled_tokens_from_alignment) == len(sim_score_per_token)
                                    # word level --> comment out for token level
                                    sim_score_per_word = fuse_align_tokens(sim_score_per_token)
                                    assert len(labeled_words_from_alignment) == len(sim_score_per_word)


                print(hard_labels, "###") #, sample['hard_labels']) #TODO: comment out when working with unlabelled test set
                if not a.sim_scores:
                    pred['soft_labels'] = [{"start":start, "prob": 1.0, "end":end} for start, end in hard_labels] # TODO: actually calculate a real prob – now i only have the dummy value 1
                else:
                    # choose granularity
                    # word level
                    #pred['soft_labels'] = [{"start":start, "prob": prob, "end":end} for (start, end), (word, prob) in zip(hard_labels, sim_score_per_word)]
                    # token level
                    pred['soft_labels'] = [{"start":start, "prob": prob, "end":end} for (start, end), (token, prob) in zip(hard_labels, sim_score_per_token)] 
                pred['hard_labels'] = hard_labels
                pred_json = json.dumps(pred, ensure_ascii=False)
                g.write(pred_json+"\n")
            print(f'\nThe file has been stored under this path:\n{outfile}')
                        # b. the other two models
                        # c. a combination of probs under different models ((maybe weighted avg))

# todo make another validation set of this validation set (half half but don't just take random splits but see how i can mitigated biases that i get from random splits)