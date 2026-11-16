# Cerebro-1000 Benchmark Evaluation Report

- **Evaluated Checkpoint:** `checkpoints/v5_small_sft_factual_calibrated.pt`
- **Evaluation Date:** 2026-11-16 12:02:44
- **Total Prompts Evaluated:** 1000
- **Overall Loss:** 7.0442 (Perplexity: 1146.14)
- **Overall Accuracy (Regex):** 3.80%
- **Overall Action Accuracy:** 19.00%

## Performance by Pillar

| Pillar Category | Evaluated | Regex Acc | Exact Acc | Action Acc | Loss | Perplexity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **linguistic_and_syntax** | 200 | 14.5% | 14.5% | 12.5% | 5.4275 | 227.57 |
| **factual_and_counterfactual** | 200 | 3.0% | 3.0% | 13.0% | 9.0529 | 8542.95 |
| **mathematical_and_logic** | 200 | 1.0% | 1.0% | 22.0% | 4.8474 | 127.41 |
| **code_and_algorithms** | 200 | 0.5% | 1.0% | 10.0% | 7.1537 | 1278.88 |
| **needle_and_agentic** | 200 | 0.0% | 0.0% | 37.5% | 8.7393 | 6243.52 |
| **TOTAL** | 1000 | **3.80%** | **3.90%** | **19.00%** | **7.0442** | **1146.14** |

