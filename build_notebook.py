"""
Generates and executes `FGAI_Assignment3.ipynb` containing the complete
implementation, mathematical foundations, dataset analysis, model evaluations,
sampling-temperature study, and ethical/responsible AI audit.
"""
import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient


def create_notebook():
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    cells = []

    # Title & Overview Markdown
    cells.append(
        new_markdown_cell(
            """# Neural Machine Translation: Comparative Analysis of Encoder–Decoder and Decoder-Only Architectures
**Course:** Foundations of Generative AI (ICT4403) — Assignment 3  
**Author:** Ayrton Joseph Pulikottil (`230953448`)  
**Language Pair:** English (`en`) $\\rightarrow$ French (`fr`)  
**GitHub Repository:** [https://github.com/Lemonseeds/gen-ai-proj-2](https://github.com/Lemonseeds/gen-ai-proj-2)  
**Dataset:** [OPUS Books Parallel Corpus (`Helsinki-NLP/opus_books`, `en-fr`)](https://huggingface.co/datasets/Helsinki-NLP/opus_books) ([OPUS Portal](https://opus.nlpl.eu/Books.php))

---

## 1. Problem Statement & Mathematical Formulation

Neural Machine Translation (NMT) models the conditional probability distribution of a target-language sequence $Y = (y_1, y_2, \\dots, y_M)$ given a source-language sequence $X = (x_1, x_2, \\dots, x_N)$.

### 1.1 Encoder–Decoder Architecture (Basic Transformer & mT5)
An encoder–decoder architecture separates bidirectional source representation from causal target generation:
$$\\mathbf{H}_{\\text{enc}} = \\text{Encoder}(x_1, \\dots, x_N) \\in \\mathbb{R}^{N \\times d_{\\text{model}}}$$
$$P_{\\theta}(Y \\mid X) = \\prod_{t=1}^{M} P_{\\theta}\\left(y_t \\;\\middle|\\; y_{<t}, \\mathbf{H}_{\\text{enc}}\\right)$$
Each decoder layer applies **masked multi-head self-attention** over $y_{<t}$ followed by **multi-head cross-attention** where queries come from the decoder state and keys/values come from $\\mathbf{H}_{\\text{enc}}$:
$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}} + M\\right)V$$

### 1.2 Decoder-Only Causal Architecture (LLaMA & Gemma)
A decoder-only causal language model processes the concatenated prompt and source sequence $C = (p_1, \\dots, p_K, x_1, \\dots, x_N)$ and target sequence $Y = (y_1, \\dots, y_M)$ within a single unified causal self-attention stack:
$$P_{\\theta}(Y \\mid C) = \\prod_{t=1}^{M} P_{\\theta}\\left(y_t \\;\\middle|\\; C, y_{<t}\\right)$$

### 1.3 Implemented Models
1. **Basic Transformer (Encoder–Decoder, Trained from Scratch):** Custom 6-layer ($3$ encoder + $3$ decoder) PyTorch Transformer (`d_model=256`, `nhead=8`, `d_ff=512`, sinusoidal positional encoding, label-smoothed cross-entropy $\\epsilon=0.1$).
2. **mT5-Small (`google/mt5-small`, Encoder–Decoder, Fine-Tuned):** Multilingual T5 pretrained via span-corruption on mC4 across 101 languages ($300.18\\text{M}$ total parameters; adapted via English–French sub-vocabulary projection and seq2seq fine-tuning).
3. **LLaMA (`LlamaForCausalLM` / `SmolLM2-135M-Instruct`, Decoder-Only):** 30-layer LLaMA architecture with Grouped-Query Attention (GQA), Rotary Positional Embeddings (RoPE), SwiGLU MLP, and RMSNorm ($134.52\\text{M}$ parameters; adapted via Few-Shot In-Context Prompting and LoRA fine-tuning $r=8, \\alpha=16$).
4. **Gemma (`Gemma3ForCausalLM` / `gemma-3-270m-it`, Decoder-Only):** 18-layer Gemma-3 architecture with QK-Norm, sliding local + global attention, GeGLU activations, and 262k SentencePiece vocabulary ($268.10\\text{M}$ parameters; adapted via 3-Shot In-Context Prompting and LoRA fine-tuning)."""
        )
    )

    # Cell 1: Imports & Dataset Inspection
    cells.append(
        new_markdown_cell(
            "## 2. Dataset Description, Preprocessing & Partitioning (`Part i`)"
        )
    )
    cells.append(
        new_code_cell(
            """import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown

from src.data import prepare_parallel_corpus, get_ethical_diagnostic_probes
from src.models import BasicTransformerSeq2Seq, MT5TranslationSystem, CausalLLMTranslationSystem

bundle = prepare_parallel_corpus(subset_size=6000, eval_test_size=50)
stats_df = pd.DataFrame(list(bundle.stats.items()), columns=["Corpus Attribute / Metric", "Empirical Value"])
display(stats_df)"""
        )
    )

    cells.append(
        new_code_cell(
            """# Display sample cleaned parallel pairs from the training split
sample_pairs_df = pd.DataFrame(bundle.train_pairs[:8])[["en", "fr"]]
sample_pairs_df.columns = ["Source Sentence (English)", "Target Reference (French)"]
display(sample_pairs_df)

# Render Dataset Overview Visualization
display(Image(filename="results/dataset_overview.png"))"""
        )
    )

    # Cell 2: Training & Adaptation Dynamics
    cells.append(
        new_markdown_cell(
            """## 3. Model Implementation, Training & Adaptation (`Part i`)
All four models are trained/adapted and evaluated on the identical held-out OPUS Books test set:
- **Basic Transformer:** Trained from scratch for 12 epochs using AdamW ($\\eta = 1.2 \\times 10^{-3}$, cosine annealing schedule, label smoothing $\\epsilon = 0.1$).
- **mT5-Small:** Fine-tuned for 5 epochs with English–French SentencePiece vocabulary projection (`5,587` active subwords out of `250,112`).
- **LLaMA (`LlamaForCausalLM`) & Gemma (`Gemma3ForCausalLM`):** Evaluated under both **3-Shot In-Context Prompting** and **LoRA Fine-Tuning** ($r=8, \\alpha=16$ applied to `q_proj, k_proj, v_proj, o_proj`)."""
        )
    )

    cells.append(
        new_code_cell(
            """hist_df = pd.read_csv("results/training_history.csv")
display(hist_df.groupby("model").tail(3))
display(Image(filename="results/training_dynamics.png"))"""
        )
    )

    # Cell 3: Quantitative Evaluation & Architecture Comparison
    cells.append(
        new_markdown_cell(
            """## 4. Quantitative Evaluation & Architectural Comparison (`Part ii`)
We evaluate all four models using **Corpus BLEU**, **Corpus chrF (chrF2)**, **Training/Fine-Tuning Wall-Clock Time**, **Inference Latency**, and **Parameter Counts** (Total vs. Trainable)."""
        )
    )

    cells.append(
        new_code_cell(
            """metrics_df = pd.read_csv("results/model_metrics.csv")
display(metrics_df)

ablation_df = pd.read_csv("results/adaptation_ablation.csv")
print("\\n--- Decoder-Only Adaptation Method Ablation (3-Shot Prompting vs. LoRA Fine-Tuning) ---")
display(ablation_df)

display(Image(filename="results/model_comparison.png"))
display(Image(filename="results/architecture_tradeoff.png"))"""
        )
    )

    # Cell 4: Side-by-Side Test Translations
    cells.append(
        new_markdown_cell(
            "### 4.1 Side-by-Side Generated Translations on the Held-Out Test Set"
        )
    )
    cells.append(
        new_code_cell(
            """samples_df = pd.read_csv("results/test_translations_comparison.csv")
display(samples_df.head(12))"""
        )
    )

    # Cell 5: Sampling-Temperature Analysis
    cells.append(
        new_markdown_cell(
            """## 5. Sampling-Temperature Analysis ($T \\in \\{0.0, 0.3, 0.7, 1.2\\}$)
To investigate the trade-off between deterministic fidelity and stochastic diversity across encoder–decoder and decoder-only paradigms, we decode the test subset across four temperatures $T \\in \\{0.0, 0.3, 0.7, 1.2\\}$."""
        )
    )
    cells.append(
        new_code_cell(
            """temp_df = pd.read_csv("results/temperature_metrics.csv")
display(temp_df[["Model", "Temperature", "BLEU", "chrF", "Distinct_1", "Distinct_2", "Repetition_Rate", "Length_Ratio"]])
display(Image(filename="results/temperature_analysis.png"))"""
        )
    )

    # Cell 6: Responsible & Ethical AI Analysis
    cells.append(
        new_markdown_cell(
            """## 6. Responsible & Ethical AI Audit in Machine Translation (`Part iii`)
We probe all four models on a curated diagnostic suite targeting five critical ethical dimensions in English $\\rightarrow$ French translation:
1. **Occupational Gender Stereotype Bias:** English gender-neutral occupational nouns (*doctor, nurse, engineer, secretary*) mapped to gendered French nouns (*le médecin / l'infirmière*, *l'ingénieur / la secrétaire*).
2. **Pronoun Gender Defaulting:** Ambiguous plural *they* defaulting to masculine plural *ils* over feminine *elles*.
3. **Cultural & Idiomatic Misrepresentation:** Literal word-for-word translation of English idioms (*raining cats and dogs*, *buy a pig in a poke*) versus culturally grounded French idioms (*il pleut des cordes*, *acheter chat en poche*).
4. **Sociolinguistic Register & Politeness (`Tu` vs. `Vous`):** Maintaining formal *vous* when addressing *Monsieur*.
5. **Safety-Critical Negation Integrity & Hallucination:** Preserving French bipartite negation (*ne ... pas*) in medical warnings (*Do not take this strong medicine with wine or alcohol*) and factual entity/date preservation."""
        )
    )
    cells.append(
        new_code_cell(
            """ethical_df = pd.read_csv("results/ethical_diagnostic_probes.csv")
display(ethical_df[["Category", "Source_EN", "Reference_FR", "Basic_Transformer", "mT5_Small", "LLaMA_135M", "Gemma_3_270M"]])"""
        )
    )

    nb.cells = cells
    with open("FGAI_Assignment3.ipynb", "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Saved FGAI_Assignment3.ipynb; executing cells...")

    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    with open("FGAI_Assignment3.ipynb", "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Successfully executed and saved FGAI_Assignment3.ipynb!")


if __name__ == "__main__":
    create_notebook()

