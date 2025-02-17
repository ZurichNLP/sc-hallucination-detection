import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import by_organizers.participant_kit_jan25.participant_kit.scorer as scorer



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ref_file', type=scorer.load_jsonl_file_to_records)
    parser.add_argument('pred_file', type=lambda fname: scorer.load_jsonl_file_to_records(fname, is_ref=False))
    parser.add_argument('output_file', type=str)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--low-res', action='store_true')
    parser.add_argument('--subset', nargs='+', default=None)
    args = parser.parse_args()

    if not args.visualize:
        model_names = set()
        model_counter = {}
        for line in args.ref_file:
            model_names.add(line['model_id'])
            if line['model_id'] not in model_counter:
                model_counter[line['model_id']] = 1
            else:
                model_counter[line['model_id']] += 1
        
        model_results = {}
        for model_name in model_names:
            ious = np.array([scorer.score_iou(r, d) for r, d in zip(args.ref_file, args.pred_file) if r['model_id'] == model_name])
            model_results[model_name] = ious.mean()
        
        with open(args.output_file, 'w') as f:
            for model_name, iou in model_results.items():
                f.write(f'{model_name}: {iou*100:.2f}, {model_counter[model_name]}\n')
                print(f'{model_name}: {iou*100:.2f}, {model_counter[model_name]}')
        print('\n\n')
    else:
        # get visualizations for the model level result
        langs = {}
        for file in os.listdir('./results/test-set-analysis'):
            file_path = os.path.join('./results/test-set-analysis', file)
            # Normalize the language code - strip any whitespace and convert to lowercase
            lang = file.split('-')[0]
            method = '-'.join(file.split('-')[1:]).strip('.txt')
            
            if args.low_res and lang not in ['fa', 'ca', 'eu', 'cs']:
                continue
            elif args.subset and lang not in args.subset:
                continue
            elif not args.low_res and lang in ['fa', 'ca', 'eu', 'cs']:
                continue
            
            # Debug print to check language codes
            print(f"Processing file: {file}, extracted lang: {lang}, method: {method}")
            
            with open(file_path, 'r') as f:
                for line in f:
                    line_split = line.split(' ')
                    model_name = line_split[0].strip(':').split('/')[-1]
                    iou = float(line_split[1].strip(',').strip())
                    sample_size = int(line_split[-1].strip())
                    if lang not in langs:
                        langs[lang] = {}
                    if model_name not in langs[lang]:
                        langs[lang][model_name] = {}
                    langs[lang][model_name][method] = (iou, sample_size)

        # Print unique languages to verify
        print("Unique languages found:", sorted(langs.keys()))
        
        # Convert data into a DataFrame for Seaborn
        data = []
        for lang, model_results in langs.items():
            # Get sample size per model (should be same across methods)
            model_sizes = {}
            for model, method_scores in model_results.items():
                sizes = [size for _, size in method_scores.values()]
                model_sizes[model] = sizes[0]  # Take first size since all should be same
                
            for model, method_scores in model_results.items():
                for method, (score, _) in method_scores.items():
                    # Map method names to display names
                    display_method = method
                    if method == 'gpt-consistency-f-median':
                        display_method = 'GPT-C'
                    elif method == 'sc-threshold-f-median':
                        display_method = 'SC'
                        
                    data.append({
                        "Language": lang.strip(),
                        "Model": model.strip(), 
                        "Method": display_method,
                        "IoU Score": score,
                        "Sample Size": model_sizes[model],
                        "Method_order": {
                            'gpt-two-step-multi': 0,
                            'gpt-consistency-f-median': 1, 
                            'sc-threshold-f-median': 2
                        }[method]
                    })

        df = pd.DataFrame(data)

        # Set Seaborn style
        sns.set_style("whitegrid")

        # Create a single figure with subplots for each language
        unique_langs = sorted(df["Language"].unique())
        n_langs = len(unique_langs)

        # Layout based on whether subset is provided
        if args.subset and len(args.subset) == 5:
            n_rows = 2  # Two rows
            n_cols = 3  # Three columns
            start_idx = 2  # Start from position 2 to center first row
        else:
            n_rows = 1
            n_cols = n_langs

        fig = plt.figure(figsize=(10*n_cols, 6*n_rows))

        # Get global min and max for y-axis
        y_min = df["IoU Score"].min()
        y_max = df["IoU Score"].max()

        for idx, lang in enumerate(unique_langs):
            if args.subset and len(args.subset) == 5:
                if idx < 2:
                    # First two plots go in first row
                    plot_idx = idx + 2  # Start from position 2
                else:
                    # Last three plots go in second row
                    plot_idx = idx + 4  # Start from position 4
            else:
                plot_idx = idx + 1
                
            ax = plt.subplot(n_rows, n_cols, plot_idx)

            # Filter data for this language
            lang_data = df[df["Language"] == lang]

            # Create grouped bar plot with ordered methods
            bars = sns.barplot(
                data=lang_data,
                x="Model",
                y="IoU Score",
                hue="Method",
                hue_order=['gpt-two-step-multi', 'GPT-C', 'SC'],
                palette="Blues",
                edgecolor="white",
                ax=ax,
            )

            # Create new x-tick labels with sample sizes
            unique_models = lang_data["Model"].unique()
            new_labels = []
            for model in unique_models:
                sample_size = lang_data[lang_data["Model"] == model]["Sample Size"].iloc[0]
                new_labels.append(f"{model}\nn={sample_size}")

            # Formatting
            ax.set_title(f"{lang}")
            ax.set_xticklabels(new_labels, rotation=45, ha='center')

            # Only show y-axis label and ticks for leftmost subplot in each row
            if plot_idx == 1:
                ax.set_ylabel("IoU Score")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            ax.set_xlabel("")

            # Set y-axis limits to be the same for all subplots
            ax.set_ylim(y_min, y_max)

            # Add legend only to the second subplot
            if plot_idx != 3:
                ax.legend([], [], frameon=False)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save figure as PNG
        plt.savefig(args.output_file, bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    main()




