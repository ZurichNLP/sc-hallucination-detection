# Token-Level Self-Consistency for Hallucination Detection

This repository contains the code accompanying our submission to the SemEval-2025 Task 3 Mu-SHROOM, the Multilingual Shared-task on Hallucinations and Related Observable Overgeneration Mistakes shared task.


## GPT4o-mini Zero-Shot Baseline

To perform the zero-shot predictions for all languages and the chosen prompts, run the following command:

```
python scripts/get_gpt4o_predictions.py direct two-step-multi
```

To evaluate the baseline for all languages and prompts, run the following command:

```
for lang in de en fr ar zh fi hi it es sv fa ca eu cs; do
    for prompt in direct two-step-multi; do
        python scripts/scorer.py data/v2_splits/val/mushroom.${lang}-val.v2.jsonl preds_v2/val/gpt4o/${prompt}/mushroom.${lang}-val.v2.jsonl results/gpt4o-prompt/${prompt}/val_scores_${lang}.txt;
    done
done
```

Further prompts are found in and can be added to the `prompts.yaml` file.


## Token-Level GPT4o-Consistency

For each model response, 20 more responses for each sample configuration are generated using gpt:

```
for split in test [val/test_jan25]; do
    for lang in de en fr ar zh fi hi it es sv; do
        python scripts/get_gpt4o_alt_res.py ${split} --langs ${lang}
    done
done
```

The original response, for which the hallucination span is to be detected, is aligned at the token-level (Sabet, 2020) with a token from each of the responses. For each token-alignment, the similarity score is calculated:

```
for split in test [val/test_jan25]; do
    for lang in de en fr ar zh fi hi it es sv; do
        python scripts/get_alignments.py data/[v2_splits/test_jan25]/${split}/mushroom.${lang}-[val.v2/tst.v1].jsonl data/alt_res_${split}[_setv2]/alt_res/${lang} f --accelerator gpu --gpt
    done
done
```

The token-level similarity scores are then aggregated in what we call a token consistency score:

```
for split in test val; do
    for lang in de en fr ar zh fi hi it es sv; do
        python scripts/get_alignments.py data/[v2_splits/test_jan25]/${split}/mushroom.${lang}-[val.v2/tst.v1].jsonl data/alt_res_${split}[_setv2]/alt_res/${lang} f --accelerator gpu --avgs --medians --gpt
    done
done
```

Inspect the token consistency scores of the hallucinated spans vs. the not-hallucinated spans in the calibration set. The threshold is chosen by maximizing the F1 score. This step is skipped for the unlabelled test set.

```
python scripts/analyze_token_sims.py [file to test set sim avgs] [--lang XY] [--fewer-altres]
```

The resulting token consistency scores are then used to detect hallucinated tokens by comparing the token consistency scores to the threshold:

```
python scripts/get_predictions.py gpt-consistency-f-median [test_jan25/val] 
```

Evaluate the predictions:

```
python scripts/scorer.py data/[v2_splits/val/test_jan25_labelled]/mushroom.${lang}-[val.v2/tst.v1].jsonl ./preds_v2/val/gpt-consistency-f-median/mushroom.${lang}-[val.v2/tst.v1].jsonl results/gpt-consistency-f-median/val_scores_${lang}.txt
```

## Token-Level Self-Consistency

The same process is repeated for the self-consistency approach using the underlying model's alternative responses:

```
for split in test val; do
    for lang in de en fr ar zh hi it es fi sv; do
        python scripts/get_alignments.py data/[v2_splits/test_jan25]/${split}/mushroom.${lang}-[val.v2/tst.v1].jsonl data/alt_res_${split}[_setv2]/alt_res/${lang} f --accelerator gpu
    done
done
```

```
for split in test val; do
    for lang in de en fr ar zh hi it es fi sv; do
        python scripts/get_alignments.py data/[v2_splits/test_jan25]/${split}/mushroom.${lang}-[val.v2/tst.v1].jsonl data/alt_res_${split}[_setv2]/alignments_f/${lang} f --avgs --medians
    done
done
```

```

Add the --less-alt-res-params flag to only include a certain number of alternative responses per sampling method (e.g.: 5 or 10) and sampling method itself (e.g.: k50_p0.90_t0.1 or all). 


```
python scripts/analyze_token_sims.py [file to test set sim avgs] [--lang XY] [--fewer-altres]
```


```
python scripts/get_predictions.py [test_jan25/val] [sc-threshold-f-median/gpt-consistency-f-median]
```

```
for lang in de en fr ar zh hi it es sv fi; do
    python scripts/scorer.py data/[v2_splits/val/test_jan25_labelled]/mushroom.${lang}-[val.v2/tst.v1].jsonl ./preds_v2/val/sc-threshold-f-median/mushroom.${lang}-[val.v2/tst.v1].jsonl results/sc-threshold-f-median/val_scores_${lang}.txt
done
```

