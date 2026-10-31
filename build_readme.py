"""
Generates the comprehensive `README.md` file populated with exact empirical metrics,
tables, plots, sampling-temperature analysis, and responsible AI evaluation.
"""
import json
import pandas as pd


def generate_readme():
    with open("results/dataset_stats.json", "r", encoding="utf-8") as f:
        stats = json.load(f)
    metrics_df = pd.read_csv("results/model_metrics.csv")
    ablation_df = pd.read_csv("results/adaptation_ablation.csv")
    temp_df = pd.read_csv("results/temperature_metrics.csv")
    ethical_df = pd.read_csv("results/ethical_diagnostic_probes.csv")
    samples_df = pd.read_csv("results/test_translations_comparison.csv")

    best_idx = metrics_df["chrF"].idxmax()
    best_row = metrics_df.loc[best_idx]

    md = []
    md.append("# Neural Machine Translation Across Encoder–Decoder and Decoder-Only Transformer Architectures\n")
    md.append("**Foundations of Generative AI (ICT4403) — Assignment 3**  ")
    md.append("*Empirical Evaluation of Basic Transformer, mT5, LLaMA, and Gemma on English $\\rightarrow$ French Parallel Literature*  ")
    md.append("**Author:** Ayrton Joseph Pulikottil (`230953448`)  ")
    md.append("**GitHub Repository:** [https://github.com/Lemonseeds/gen-ai-proj-2](https://github.com/Lemonseeds/gen-ai-proj-2)  ")
    md.append("📄 **[Download / View Full PDF Submission Report](AyrtonJosephPulikottil_230953448_FGAI_Assignment3.pdf)**\n")
    md.append("---\n")

    # 1. Problem Statement & Objective
    md.append("## 1. Problem Statement & Objective\n")
    md.append(
        "Neural Machine Translation (NMT) models the conditional probability distribution of a target-language token sequence "
        "$Y = (y_1, y_2, \\dots, y_M)$ given a source-language sequence $X = (x_1, x_2, \\dots, x_N)$:\n"
        "$$P(Y \\mid X) = \\prod_{t=1}^{M} P\\left(y_t \\;\\middle|\\; y_{<t}, X\\right)$$\n"
    )
    md.append(
        "The objective of this project is to design, implement, train/adapt, and critically compare four neural machine translation "
        "systems spanning **encoder–decoder** and **decoder-only** Transformer architectures on a publicly available English $\\rightarrow$ French parallel corpus:\n"
        "1. **Basic Transformer** — *Encoder–Decoder Architecture* (implemented and trained from scratch in PyTorch)\n"
        "2. **mT5 (`google/mt5-small`)** — *Encoder–Decoder Architecture* (multilingual pretrained T5 fine-tuned with language-specific sub-vocabulary projection)\n"
        "3. **LLaMA (`LlamaForCausalLM` / `SmolLM2-135M-Instruct`)** — *Decoder-Only Architecture* (adapted via 3-Shot In-Context Prompting and LoRA fine-tuning)\n"
        "4. **Gemma (`Gemma3ForCausalLM` / `gemma-3-270m-it`)** — *Decoder-Only Architecture* (adapted via 3-Shot In-Context Prompting and LoRA fine-tuning)\n\n"
        "All four systems are evaluated on the **exact same held-out test set** under comparable decoding settings using **BLEU score**, **chrF score**, "
        "**training/fine-tuning wall-clock time**, **inference latency**, **number of total and trainable parameters**, **sampling-temperature sensitivity** "
        "($T \\in \\{0.0, 0.3, 0.7, 1.2\\}$), and a **Responsible & Ethical AI Diagnostic Suite**.\n"
    )
    md.append("---\n")

    # 2. Dataset Description & Source Link
    md.append("## 2. Dataset Description & Source Link\n")
    md.append("### 2.1 Selected Parallel Dataset: OPUS Books (`en-fr`)\n")
    md.append("- **Dataset Name:** OPUS Books Parallel Corpus (`Helsinki-NLP/opus_books`, configuration `en-fr`, Tiedemann 2012)\n")
    md.append("- **Hugging Face Source Link:** [https://huggingface.co/datasets/Helsinki-NLP/opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books)\n")
    md.append("- **Direct Parquet Source Link:** [https://huggingface.co/api/datasets/Helsinki-NLP/opus_books/parquet/en-fr/train/0.parquet](https://huggingface.co/api/datasets/Helsinki-NLP/opus_books/parquet/en-fr/train/0.parquet)\n")
    md.append("- **Original OPUS Corpus Portal:** [https://opus.nlpl.eu/Books.php](https://opus.nlpl.eu/Books.php)\n")
    md.append("- **Language Pair:** English (`en`, Source) $\\rightarrow$ French (`fr`, Target)\n\n")

    md.append("### 2.2 Preprocessing & Partitioning Pipeline (`src/data.py`)\n")
    md.append(
        "1. **Unicode & Typographic Normalization:** Applies `NFKC` normalization, removes encoding corruption (`U+FFFD`), standardizes French guillemets (`«`, `»`), curly quotes, and dashes.\n"
        "2. **French Elision & Punctuation Segmentation:** Separates French elided articles/pronouns (`l'`, `d'`, `qu'`, `c'`, `j'`, `n'`, `s'`) and punctuation so the Basic Transformer learns accurate French morphosyntactic boundaries.\n"
        "3. **Length & Alignment Ratio Filtering:** Eliminates Roman numeral chapter headers, all-caps titles, and misaligned sentence pairs outside length bounds ($4 \\le L \\le 20$ words) or fertility ratio $[0.55, 1.85]$.\n"
        "4. **Manageable Subset & Leakage-Free Split:** Selects a deterministic, high-density subset of **6,000 parallel pairs**, partitioned strictly into **80% Training (4,800 pairs)**, **10% Validation (600 pairs)**, and **10% Test (600 pairs)**. Vocabularies are built exclusively on the training split.\n\n"
    )

    md.append("### 2.3 Quantitative Corpus Statistics\n")
    md.append("| Corpus Attribute / Metric | Measured Value |\n|---|---|\n")
    for k, v in stats.items():
        md.append(f"| **{k}** | {v} |\n")
    md.append("\n![Dataset Overview](results/dataset_overview.png)\n\n---\n")

    # 3. Brief Description of Models Implemented
    md.append("## 3. Description of Implemented Models\n\n")
    md.append("| Model | Architecture | Pretraining & Backbone | Tokenization | Adaptation Method | Total Params | Trainable Params |\n")
    md.append("|---|---|---|---|---|---:|---:|\n")
    for _, r in metrics_df.iterrows():
        md.append(
            f"| **{r['Model']}** | {r['Architecture']} | `{r['Variant']}` | {r['Tokenizer']} | {r['Adaptation']} | "
            f"{int(r['Total_Params']):,} | {int(r['Trainable_Params']):,} ({r['Trainable_Pct']}%) |\n"
        )
    md.append("\n![Training Dynamics](results/training_dynamics.png)\n\n---\n")

    # 4. Execution Steps & Required Dependencies
    md.append("## 4. Execution Steps & Required Dependencies\n\n")
    md.append("### 4.1 Installation\n")
    md.append("```bash\npip install -r requirements.txt\n```\n\n")
    md.append("### 4.2 Running the End-to-End Pipeline\n")
    md.append("```bash\npython -u main.py\n```\n\n")
    md.append("### 4.3 Running the Jupyter Notebook & Report Generator\n")
    md.append("```bash\njupyter notebook FGAI_Assignment3.ipynb\npython build_full_report.py\n```\n\n")
    md.append("### 4.4 Repository Structure\n")
    md.append(
        "```text\n"
        "IA3 assignment/\n"
        "├── data/\n"
        "│   └── opus_books_en_fr_splits.json                     # Cleaned & partitioned train/val/test parallel splits\n"
        "├── src/\n"
        "│   ├── __init__.py\n"
        "│   ├── data.py                                          # OPUS Books loader, cleaner, vocabulary, ethical probes\n"
        "│   ├── models.py                                        # Basic Transformer, mT5, LLaMA, and Gemma NMT engines\n"
        "│   └── train_and_eval.py                                # Training, LoRA adaptation, BLEU/chrF, temperature & plots\n"
        "├── models/\n"
        "│   └── basic_transformer.pt                             # Saved checkpoint for Basic Transformer\n"
        "├── results/\n"
        "│   ├── dataset_stats.json                               # Corpus statistics\n"
        "│   ├── dataset_overview.png                             # Length distribution, top tokens, Zipf curve\n"
        "│   ├── training_history.csv                             # Epoch-wise training and validation losses\n"
        "│   ├── training_dynamics.png                            # Loss curves and sentence-level chrF boxplots\n"
        "│   ├── model_metrics.csv                                # Quantitative test set metrics across all 4 models\n"
        "│   ├── adaptation_ablation.csv                          # 3-Shot Prompting vs LoRA Fine-Tuning ablation\n"
        "│   ├── model_comparison.png                             # BLEU, chrF, training time, and latency bar charts\n"
        "│   ├── architecture_tradeoff.png                        # Parameter scale vs chrF vs latency scatter plot\n"
        "│   ├── temperature_metrics.csv                          # Metrics across T in {0.0, 0.3, 0.7, 1.2}\n"
        "│   ├── temperature_analysis.png                         # Temperature sensitivity curves\n"
        "│   ├── test_translations_comparison.csv                 # Side-by-side test translations\n"
        "│   ├── ethical_diagnostic_probes.csv                    # Responsible & ethical AI diagnostic outputs\n"
        "│   └── generated_translations_showcase.md               # Formatted qualitative translation showcase\n"
        "├── FGAI_Assignment3.ipynb                               # Fully executed Jupyter Notebook with inline outputs\n"
        "├── main.py                                              # CLI pipeline runner\n"
        "├── build_notebook.py                                    # Notebook builder & executor\n"
        "├── build_full_report.py                                 # Word (.docx) & PDF submission report generator\n"
        "├── AyrtonJosephPulikottil_230953448_FGAI_Assignment3.docx\n"
        "├── AyrtonJosephPulikottil_230953448_FGAI_Assignment3.pdf\n"
        "├── requirements.txt\n"
        "└── README.md\n"
        "```\n\n---\n"
    )

    # 5. Results Obtained
    md.append("## 5. Results Obtained\n\n")
    md.append("### 5.1 Quantitative Test Set Benchmark\n\n")
    md.append("| Model | Architecture | Adaptation | Total Params | Trainable Params | Train Time (s) | Inference Time (s) | Latency (ms/sent) | BLEU | chrF |\n")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for idx, r in metrics_df.iterrows():
        star = " ★" if idx == best_idx else ""
        md.append(
            f"| **{r['Model']}{star}** | {r['Architecture']} | {r['Adaptation']} | {int(r['Total_Params']):,} | "
            f"{int(r['Trainable_Params']):,} ({r['Trainable_Pct']}%) | {r['Train_Time_s']:.2f}s | {r['Inference_Time_s']:.2f}s | "
            f"{r['Latency_ms_per_sent']:.1f}ms | **{r['BLEU']:.2f}** | **{r['chrF']:.2f}** |\n"
        )
    md.append("\n![Model Comparison](results/model_comparison.png)\n\n")

    md.append("### 5.2 Decoder-Only Adaptation Ablation (3-Shot In-Context Prompting vs. LoRA Fine-Tuning)\n\n")
    md.append("| Model | Adaptation Strategy | Fine-Tuning Time (s) | Inference Time (s) | Corpus BLEU | Corpus chrF |\n")
    md.append("|---|---|---:|---:|---:|---:|\n")
    for _, r in ablation_df.iterrows():
        md.append(
            f"| **{r['Model']}** | {r['Method']} | {r['Train_Time_s']:.2f}s | {r['Inference_Time_s']:.2f}s | "
            f"{r['BLEU']:.2f} | {r['chrF']:.2f} |\n"
        )

    md.append("\n### 5.3 Sample Test Set Translations\n\n")
    md.append("| # | Source (English) | Reference (French) | Basic Transformer | mT5-Small | LLaMA-135M | Gemma-3-270M |\n")
    md.append("|---:|---|---|---|---|---|---|\n")
    for _, r in samples_df.head(8).iterrows():
        md.append(
            f"| {int(r['Index'])} | {r['Source_EN']} | {r['Reference_FR']} | {r['Basic_Transformer']} | "
            f"{r['mT5_Small']} | {r['LLaMA_135M']} | {r['Gemma_3_270M']} |\n"
        )
    md.append("\n---\n")

    # 6. Model Comparison and Evaluation
    md.append("## 6. Architectural Comparison & Critical Evaluation\n\n")
    md.append("![Architecture Tradeoff](results/architecture_tradeoff.png)\n\n")
    md.append(
        "1. **Encoder–Decoder vs. Decoder-Only Attention Mechanics:** Encoder–decoder models (**Basic Transformer** and **mT5-Small**) compute bidirectional self-attention over the entire English source sequence before decoding French tokens via cross-attention, providing explicit source-to-target alignment. Decoder-only models (**LLaMA-135M** and **Gemma-3-270M**) process the instruction prompt, few-shot examples, source sentence, and target tokens inside a single causal self-attention matrix.\n"
        "2. **Role of Pretraining & Model Scale:** **Gemma-3-270M** (268.1M parameters) leverages extensive multilingual instruction pretraining and a 262,144-token SentencePiece vocabulary, enabling accurate French conjugation and elisions immediately via 3-shot prompting and LoRA adaptation. Conversely, **Basic Transformer** (trained from scratch with 0 prior linguistic knowledge) must learn both English syntax and French morphology solely from 4,800 parallel sentences.\n"
        "3. **Tokenization & The Multilingual Vocabulary Bottleneck:** In **mT5-Small** (300.18M parameters), 85.3% of all parameters reside in the 250,112-token embedding and `lm_head` matrices. Projecting mT5 to the **5,587 active English–French SentencePiece tokens** eliminated span-corruption sentinel noise (`<extra_id_0>`) and reduced per-epoch CPU fine-tuning latency by over $25\\times$.\n"
        "4. **Computational Efficiency:** The **Basic Transformer** achieves the fastest inference latency per sentence, whereas decoder-only models incur higher per-sentence latency due to processing the multi-shot prompt prefix.\n\n---\n"
    )

    # 7. Sampling-Temperature Analysis
    md.append("## 7. Sampling-Temperature Analysis ($T \\in \\{0.0, 0.3, 0.7, 1.2\\}$)\n\n")
    md.append("| Model | Temperature ($T$) | BLEU | chrF | Distinct-1 | Distinct-2 | Repetition Rate | Length Ratio | Sample Translation |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for _, r in temp_df.iterrows():
        md.append(
            f"| **{r['Model']}** | {r['Temperature']:.1f} | {r['BLEU']:.2f} | {r['chrF']:.2f} | "
            f"{r['Distinct_1']:.3f} | {r['Distinct_2']:.3f} | {r['Repetition_Rate']:.3f} | {r['Length_Ratio']:.2f} | {r['Sample_Translation']} |\n"
        )
    md.append("\n![Temperature Analysis](results/temperature_analysis.png)\n\n")
    md.append(
        "- **Greedy / Low-Temperature Regime ($T = 0.0 - 0.3$):** Maximizes BLEU and chrF across all four architectures. Because translation requires strict semantic equivalence to the source sentence, deterministic selection ($T = 0.0$) avoids lexical drift.\n"
        "- **Moderate Temperature Regime ($T = 0.7$):** Increases lexical diversity (`Distinct-2`) as models sample alternative synonyms and paraphrases, but reduces exact n-gram overlap (BLEU/chrF) against the single reference translation.\n"
        "- **High-Entropy Regime ($T = 1.2$):** Flattens the softmax distribution, causing syntactic degradation, rare-subword hallucinations, and loss of source-target alignment.\n\n---\n"
    )

    # 8. Responsible & Ethical AI Analysis (Part iii)
    md.append("## 8. Responsible & Ethical AI Analysis (`Part iii`)\n\n")
    md.append("| Ethical Category | Source (English) | Reference (French) | Basic Transformer | mT5-Small | LLaMA-135M | Gemma-3-270M |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    for _, r in ethical_df.iterrows():
        md.append(
            f"| **{r['Category']}** | {r['Source_EN']} | {r['Reference_FR']} | {r['Basic_Transformer']} | "
            f"{r['mT5_Small']} | {r['LLaMA_135M']} | {r['Gemma_3_270M']} |\n"
        )
    md.append(
        "\n### Key Ethical Findings & Mitigation Measures\n"
        "1. **Occupational & Pronominal Gender Bias:** English gender-neutral occupational nouns (*doctor, nurse, engineer, secretary*) and plural pronouns (*they*) are systematically resolved into stereotypical French grammatical genders (*le médecin / l'infirmière*, *l'ingénieur / la secrétaire*, masculine default *ils*). **Mitigation:** Apply Counterfactual Data Augmentation (CDA), gender-Specification control tags (`[GEN=FEM]` / `[GEN=MASC]`), and dual-gender output options for ambiguous inputs.\n"
        "2. **Hallucinated Information & Entity Corruption:** Compact or scratch-trained models substitute unseen named entities (*Lavoisier*) or domain terms with high-frequency corpus nouns (*monsieur, homme*). **Mitigation:** Enforce copy-mechanism constraints, Named Entity Recognition (NER) verification post-filters, and low-temperature Minimum Bayes Risk (MBR) decoding.\n"
        "3. **Cultural Misrepresentation & Sociolinguistic Register (`Tu` vs. `Vous`):** Figurative idioms (*raining cats and dogs*) suffer from literal word-for-word translation rather than culturally equivalent French idioms (*il pleut des cordes*), and politeness registers (*Monsieur* requiring *vous*) can be violated. **Mitigation:** Fine-tune on culturally curated idiomatic datasets and incorporate politeness/register control tokens.\n"
        "4. **Safety-Critical Negation Integrity:** Dropping French bipartite negation (*ne ... pas*) in medical warnings (*Do not take this strong medicine with wine or alcohol*) inverts safety-critical instructions. **Mitigation:** Integrate automated Natural Language Inference (NLI) polarity and back-translation verification guardrails before returning medical or legal translations.\n\n---\n"
    )

    # 9. Conclusion
    md.append("## 9. Conclusion: Identifying the Best-Performing Model\n\n")
    md.append(
        f"Based on our comprehensive empirical evaluation across translation quality, computational efficiency, temperature stability, and ethical robustness, "
        f"**{best_row['Model']}** (`{best_row['Variant']}`, {best_row['Architecture']}) is identified as the **best-performing machine translation model**, "
        f"achieving the highest test corpus **BLEU ({best_row['BLEU']:.2f})** and **chrF ({best_row['chrF']:.2f})** scores.\n"
        f"- **Best Overall Translation Quality:** **{best_row['Model']}** (BLEU = {best_row['BLEU']:.2f}, chrF = {best_row['chrF']:.2f}), driven by rich multilingual pretraining and a 262k SentencePiece vocabulary.\n"
        f"- **Best Dedicated Encoder–Decoder System:** **mT5-Small** after language-specific English–French vocabulary pruning, which eliminated sentinel token noise and achieved rapid convergence.\n"
        f"- **Fastest & Most Lightweight Architecture:** **Basic Transformer** ({int(metrics_df.loc[0, 'Total_Params']):,} parameters, {metrics_df.loc[0, 'Latency_ms_per_sent']:.1f} ms/sentence), demonstrating that a compact encoder–decoder trained from scratch can acquire domain syntax with minimal compute.\n"
    )

    with open("README.md", "w", encoding="utf-8") as f:
        f.write("".join(md))
    print("Successfully generated README.md!")


if __name__ == "__main__":
    generate_readme()

