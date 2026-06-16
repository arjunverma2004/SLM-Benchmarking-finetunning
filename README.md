# SLM Benchmarking & Fine-Tuning for Agile Project Management

Benchmarking small language models (SLMs) on Agile project management tasks, then fine-tuning the best performer with QLoRA to close the gap with larger models — at a fraction of the compute cost.

## Overview

This project evaluates how well small (8–9B parameter) language models can act as a project management assistant, generating structured outputs for four core PM tasks: task design, task allocation, sprint planning, and resource planning. Three open-weight SLMs were benchmarked head-to-head, and the strongest base model (Qwen3-8B) was then fine-tuned with QLoRA on a synthetic PM dataset to measure how much targeted fine-tuning improves task-specific performance.

**Pipeline:**

1. **Synthetic dataset generation** — a hybrid dataset combining real seed issues (Apache/Spring Jira) with synthetic samples generated via Qwen2.5-7B-Instruct and Gemma-3, covering the four PM task types.
2. **Benchmarking** — Qwen3-8B, Llama-3.1-8B-Instruct, and GLM-Z1-9B evaluated on 500 held-out samples each, scored with an LLM-as-judge plus automated metrics (BERTScore, ROUGE-L, schema/JSON validity, hallucination detection).
3. **QLoRA fine-tuning** — Qwen3-8B fine-tuned (r=16, alpha=32) on the PM dataset and re-evaluated under the same benchmark to quantify improvement.

## Repository structure

```
.
├── Benchmarking/
│   ├── GLM-Z1-9B_inference.json
│   ├── Llama-3.1-8B_001.json
│   ├── Qwen3-8B_001.json
│   └── with-results-modal_slm_p....ipynb
├── Finetunning/
│   ├── finetuned_results.json
│   └── with-results-qwen-finetun....ipynb
└── Synthetic_Dataset/
    ├── Aadi_inception/
    ├── Arj_Incep2/
    ├── Arj_inception/
    ├── Gemma/
    ├── Inception_pankaj/
    └── sprint_n_resource/
```

- **`Benchmarking/`** — inference outputs and notebooks for running all three base SLMs against the 500-sample benchmark set.
- **`Finetunning/`** — QLoRA fine-tuning notebook and the resulting evaluation outputs for the fine-tuned Qwen3-8B variant.
- **`Synthetic_Dataset/`** — generation runs for the hybrid synthetic dataset, split by generation batch/source.

## Models evaluated

| Model | Type | Quantization |
|---|---|---|
| Qwen3-8B | Base | 4-bit |
| Llama-3.1-8B-Instruct | Base | 4-bit |
| GLM-Z1-9B | Base (reasoning) | 4-bit |
| Qwen3-8B-PM-LoRA | QLoRA fine-tuned | 4-bit |

Fine-tuned LoRA adapter: **[arjunverma2004/Qwen3-8B-PM-LoRA](https://huggingface.co/arjunverma2004/Qwen3-8B-PM-LoRA)** on Hugging Face.

## Evaluation methodology

Each model was run on 500 samples spanning task design, task allocation, sprint planning, and resource planning. Outputs were scored using:

- **LLM-as-judge** — overall quality, clarity, completeness, relevance, feasibility, and Agile alignment (1–5 scale).
- **Automated metrics** — BERTScore F1, ROUGE-L, JSON validity, schema compliance, and hallucination rate.
- **Composite score** — a weighted combination of the above, used as the primary ranking metric.

## Results

Aggregate scores across all 500 evaluated samples per model (470 for the fine-tuned variant):

| Metric | GLM-Z1-9B | Llama-3.1-8B | Qwen3-8B | Qwen3-8B-PM-LoRA |
|---|---|---|---|---|
| **Composite score** | 0.426 | 0.582 | 0.586 | **0.651** |
| BERTScore F1 | 0.775 | 0.908 | 0.911 | **0.929** |
| ROUGE-L | 0.159 | 0.427 | 0.443 | **0.536** |
| Schema compliance | 0.239 | 0.438 | 0.426 | **0.454** |
| Valid JSON rate | 0.378 | **0.970** | 0.908 | 0.881 |
| Hallucination rate | 0.000 | 0.000 | 0.000 | 0.000 |
| Judge — overall (1–5) | 1.58 | 2.38 | 2.31 | **3.47** |
| Judge — clarity | 1.49 | 2.43 | 2.19 | **3.64** |
| Judge — completeness | 1.12 | 1.80 | 1.64 | **3.17** |
| Judge — feasibility | 1.71 | 2.56 | 2.56 | **3.46** |
| Judge — Agile alignment | 1.53 | 2.28 | 2.23 | **3.49** |
| Avg. latency (s) | 7.31 | 3.29 | 3.91 | **0.93** |
| Throughput (tokens/s) | 77.6 | 97.3 | 89.0 | **445.9** |

<img width="1484" height="733" alt="compare_composite_bar" src="https://github.com/user-attachments/assets/7a6e7216-6dd4-4122-9665-5976753f0dd1" />


**Key findings:**

- QLoRA fine-tuning improved Qwen3-8B's composite score from 0.586 to 0.651 and roughly doubled judge-rated completeness and clarity, while cutting average latency by over 4x and increasing throughput by ~5x — largely because the fine-tuned model no longer wastes tokens on the verbose reasoning traces that the base model produces.
- GLM-Z1-9B underperformed across the board, primarily because its thinking-model architecture frequently exhausted its token budget before producing a final answer, resulting in low JSON validity (37.8%) and the highest latency of the three base models.
- Llama-3.1-8B had the highest raw JSON validity among base models (97%) but lagged Qwen3-8B and the fine-tuned variant on judge-rated quality dimensions.
- Composite scores by task type consistently favored the fine-tuned model across all four task categories (task design, task allocation, sprint planning, resource planning), with the largest relative gains on resource planning and sprint planning.

Full per-sample results are available in [`all_models_results.csv`](./all_models_results.csv).

## Tech stack

- **Inference:** Modal Lab's L40S GPUs for benchmarking, A100-40 Gb GPU for fine-tuning
- **Fine-tuning:** QLoRA (4-bit, r=16, alpha=32) via SFTTrainer
- **Evaluation:** LLM-as-judge (Mercury-2 / Inception API), BERTScore, ROUGE-L, custom JSON schema validators
- **Dataset:** Hybrid synthetic generation (Mercury-2, Gemma-3) + real Jira seed data

## Links

- LoRA adapter: [huggingface.co/arjunverma2004/Qwen3-8B-PM-LoRA](https://huggingface.co/arjunverma2004/Qwen3-8B-PM-LoRA)
