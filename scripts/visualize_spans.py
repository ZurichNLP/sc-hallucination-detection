from termcolor import colored
import sys
import json

def highlight_spans(text, soft_labels):
    highlighted_text = ""
    prev_end = 0
    for label in soft_labels:
        start = label["start"]
        end = label["end"]
        prob = label["prob"]
        
        if prob >= 0.8:
            color = 'blue'
            attrs = []
        elif prob >= 0.5:
            color = 'magenta'
            attrs = []
        else:
            color = 'cyan'
            attrs = []
        
        highlighted_text += text[prev_end:start]
        
        highlighted_text += colored(text[start:end], color, attrs=attrs)
        
        prev_end = end
    
    highlighted_text += text[prev_end:]
    
    return highlighted_text

def print_formatted_data(json_data, json_data_pred=None):
    sample = json_data
    sample_pred = json_data_pred
    print("Model ID: ", sample["model_id"])
    print("Input: ", sample["model_input"])
    
    model_output = highlight_spans(sample["model_output_text"], sample["soft_labels"])
    print("Gold: ", model_output)
    if json_data_pred:
        model_output_pred = highlight_spans(sample_pred["model_output_text"], sample_pred["soft_labels"])
        print()
        print("Pred: ", model_output_pred)
    
    print("="*80)

file_path = sys.argv[1] # if comparing two files this should be the gold data
print(len(sys.argv))
if len(sys.argv)>2:
    file_path_pred = sys.argv[2] # this should be the predicted data
else: None
with open(file_path, 'r', encoding='utf-8') as f:
    if len(sys.argv)>2:
        with open(file_path_pred, 'r', encoding='utf-8') as p:
            for line, line_p in zip(f, p):
                print_formatted_data(json.loads(line), json_data_pred=json.loads(line_p))
    else:
        for line in f:
            print_formatted_data(json.loads(line))
