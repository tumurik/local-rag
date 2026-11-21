### Scores (source: eval_results_local.json, eval_results_cloud.json)

| Metric | n | Local (mean ± 2·SE) | Cloud (mean ± 2·SE) | Cloud − local, per question | Verdict |
|---|---|---|---|---|---|
| Answer Relevancy | 20 | 1.00 ± 0.00 | 0.96 ± 0.07 | -0.04 ± 0.07 | no clear difference |
| Contextual Precision | 20 | 0.82 ± 0.11 | 0.84 ± 0.10 | +0.02 ± 0.05 | no clear difference |
| Contextual Recall | 20 | 0.81 ± 0.12 | 0.81 ± 0.12 | -0.01 ± 0.07 | no clear difference |
| Contextual Relevancy | 20 | 0.19 ± 0.06 | 0.21 ± 0.07 | +0.02 ± 0.03 | no clear difference |
| Faithfulness | 20 | 0.84 ± 0.14 | 0.78 ± 0.17 | -0.06 ± 0.13 | no clear difference |

Verdict rule: 'no clear difference' when the per-question range (mean ± 2·SE) includes zero.

### Time, tokens, failures (source: pipeline_outputs_local.json, pipeline_outputs_cloud.json)

| Run | Model | Questions | Median generation (s) | Slowest (s) | Avg completion tokens | Empty / failed |
|---|---|---|---|---|---|---|
| local | gemma4:e4b | 20 | 10.7 | 30.3 | 504 | 0 |
| cloud | gpt-5.6-luna | 20 | 1.2 | 2.4 | 57 | 0 |

Questions where |cloud − local| ≥ 0.3 (read the judge's reasons for these):

- Answer Relevancy: q12 (-0.75)
- Contextual Precision: q20 (+0.41)
- Contextual Recall: q10 (-0.33), q12 (-0.33)
- Faithfulness: q01 (-1.00), q17 (+0.60), q05 (-0.55), q16 (-0.33)
