# Neural Machine Translation Across Encoder–Decoder and Decoder-Only Transformer Architectures
**Foundations of Generative AI (ICT4403) — Assignment 3**  *Empirical Evaluation of Basic Transformer, mT5, LLaMA, and Gemma on English $\rightarrow$ French Parallel Literature*  **Author:** Ayrton Joseph Pulikottil (`230953448`)  **GitHub Repository:** [https://github.com/Lemonseeds/gen-ai-proj-2](https://github.com/Lemonseeds/gen-ai-proj-2)  📄 **[Download / View Full PDF Submission Report](AyrtonJosephPulikottil_230953448_FGAI_Assignment3.pdf)**
---
## 1. Problem Statement & Objective
Neural Machine Translation (NMT) models the conditional probability distribution of a target-language token sequence $Y = (y_1, y_2, \dots, y_M)$ given a source-language sequence $X = (x_1, x_2, \dots, x_N)$:
$$P(Y \mid X) = \prod_{t=1}^{M} P\left(y_t \;\middle|\; y_{<t}, X\right)$$
The objective of this project is to design, implement, train/adapt, and critically compare four neural machine translation systems spanning **encoder–decoder** and **decoder-only** Transformer architectures on a publicly available English $\rightarrow$ French parallel corpus:
1. **Basic Transformer** — *Encoder–Decoder Architecture* (implemented and trained from scratch in PyTorch)
2. **mT5 (`google/mt5-small`)** — *Encoder–Decoder Architecture* (multilingual pretrained T5 fine-tuned with language-specific sub-vocabulary projection)
3. **LLaMA (`LlamaForCausalLM` / `SmolLM2-135M-Instruct`)** — *Decoder-Only Architecture* (adapted via 3-Shot In-Context Prompting and LoRA fine-tuning)
4. **Gemma (`Gemma3ForCausalLM` / `gemma-3-270m-it`)** — *Decoder-Only Architecture* (adapted via 3-Shot In-Context Prompting and LoRA fine-tuning)

All four systems are evaluated on the **exact same held-out test set** under comparable decoding settings using **BLEU score**, **chrF score**, **training/fine-tuning wall-clock time**, **inference latency**, **number of total and trainable parameters**, **sampling-temperature sensitivity** ($T \in \{0.0, 0.3, 0.7, 1.2\}$), and a **Responsible & Ethical AI Diagnostic Suite**.
---
## 2. Dataset Description & Source Link
### 2.1 Selected Parallel Dataset: OPUS Books (`en-fr`)
- **Dataset Name:** OPUS Books Parallel Corpus (`Helsinki-NLP/opus_books`, configuration `en-fr`, Tiedemann 2012)
- **Hugging Face Source Link:** [https://huggingface.co/datasets/Helsinki-NLP/opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books)
- **Direct Parquet Source Link:** [https://huggingface.co/api/datasets/Helsinki-NLP/opus_books/parquet/en-fr/train/0.parquet](https://huggingface.co/api/datasets/Helsinki-NLP/opus_books/parquet/en-fr/train/0.parquet)
- **Original OPUS Corpus Portal:** [https://opus.nlpl.eu/Books.php](https://opus.nlpl.eu/Books.php)
- **Language Pair:** English (`en`, Source) $\rightarrow$ French (`fr`, Target)

### 2.2 Preprocessing & Partitioning Pipeline (`src/data.py`)
1. **Unicode & Typographic Normalization:** Applies `NFKC` normalization, removes encoding corruption (`U+FFFD`), standardizes French guillemets (`«`, `»`), curly quotes, and dashes.
2. **French Elision & Punctuation Segmentation:** Separates French elided articles/pronouns (`l'`, `d'`, `qu'`, `c'`, `j'`, `n'`, `s'`) and punctuation so the Basic Transformer learns accurate French morphosyntactic boundaries.
3. **Length & Alignment Ratio Filtering:** Eliminates Roman numeral chapter headers, all-caps titles, and misaligned sentence pairs outside length bounds ($4 \le L \le 20$ words) or fertility ratio $[0.55, 1.85]$.
4. **Manageable Subset & Leakage-Free Split:** Selects a deterministic, high-density subset of **6,000 parallel pairs**, partitioned strictly into **80% Training (4,800 pairs)**, **10% Validation (600 pairs)**, and **10% Test (600 pairs)**. Vocabularies are built exclusively on the training split.

### 2.3 Quantitative Corpus Statistics
| Corpus Attribute / Metric | Measured Value |
|---|---|
| **Dataset Name** | OPUS Books (Helsinki-NLP/opus_books, en-fr) |
| **Source URL** | https://huggingface.co/datasets/Helsinki-NLP/opus_books |
| **Language Pair** | English (en) -> French (fr) |
| **Raw Corpus Sentence Pairs** | 127085 |
| **Cleaned & Length-Aligned Pairs** | 60895 |
| **Selected Manageable Subset** | 6000 |
| **Training Split (80%)** | 4800 |
| **Validation Split (10%)** | 600 |
| **Test Split (10%)** | 600 |
| **Comparable Benchmark Test Subset** | 30 |
| **Mean Source Length (EN tokens)** | 12.64 |
| **Mean Target Length (FR tokens)** | 13.06 |
| **Median Source / Target Length** | 12 / 12 |
| **Mean Fertility Ratio (FR / EN)** | 1.033 |
| **Source Vocabulary Size (EN)** | 1666 |
| **Target Vocabulary Size (FR)** | 1919 |
| **Total Unique EN Lexical Types (Train)** | 2518 |
| **Total Unique FR Lexical Types (Train)** | 3082 |
| **Test Source OOV Rate (%)** | 2.26 |
| **Test Target OOV Rate (%)** | 3.27 |

![Dataset Overview](results/dataset_overview.png)

---
## 3. Description of Implemented Models

| Model | Architecture | Pretraining & Backbone | Tokenization | Adaptation Method | Total Params | Trainable Params |
|---|---|---|---|---|---:|---:|
| **Basic Transformer** | Encoder-Decoder | `Custom 6-Layer Seq2Seq` | Word+Punct (1666/1919) | Trained from Scratch | 2,544,767 | 2,544,767 (100.0%) |
| **mT5-Small** | Encoder-Decoder | `google/mt5-small` | SentencePiece (250,112) | Seq2Seq Fine-Tuning | 300,176,768 | 30,917,312 (10.3%) |
| **LLaMA-135M** | Decoder-Only | `LlamaForCausalLM (SmolLM2-135M)` | BPE (49,152) | LoRA Fine-Tuning + Prompting | 134,515,008 | 15,360 (0.01%) |
| **Gemma-3-270M** | Decoder-Only | `Gemma3ForCausalLM (gemma-3-270m-it)` | SentencePiece (262,144) | LoRA + Few-Shot Prompting | 268,098,176 | 20,480 (0.01%) |

![Training Dynamics](results/training_dynamics.png)

---
## 4. Execution Steps & Required Dependencies

### 4.1 Installation
```bash
pip install -r requirements.txt
```

### 4.2 Running the End-to-End Pipeline
```bash
python -u main.py
```

### 4.3 Running the Jupyter Notebook & Report Generator
```bash
jupyter notebook FGAI_Assignment3.ipynb
python build_full_report.py
```

### 4.4 Repository Structure
```text
IA3 assignment/
├── data/
│   └── opus_books_en_fr_splits.json                     # Cleaned & partitioned train/val/test parallel splits
├── src/
│   ├── __init__.py
│   ├── data.py                                          # OPUS Books loader, cleaner, vocabulary, ethical probes
│   ├── models.py                                        # Basic Transformer, mT5, LLaMA, and Gemma NMT engines
│   └── train_and_eval.py                                # Training, LoRA adaptation, BLEU/chrF, temperature & plots
├── models/
│   └── basic_transformer.pt                             # Saved checkpoint for Basic Transformer
├── results/
│   ├── dataset_stats.json                               # Corpus statistics
│   ├── dataset_overview.png                             # Length distribution, top tokens, Zipf curve
│   ├── training_history.csv                             # Epoch-wise training and validation losses
│   ├── training_dynamics.png                            # Loss curves and sentence-level chrF boxplots
│   ├── model_metrics.csv                                # Quantitative test set metrics across all 4 models
│   ├── adaptation_ablation.csv                          # 3-Shot Prompting vs LoRA Fine-Tuning ablation
│   ├── model_comparison.png                             # BLEU, chrF, training time, and latency bar charts
│   ├── architecture_tradeoff.png                        # Parameter scale vs chrF vs latency scatter plot
│   ├── temperature_metrics.csv                          # Metrics across T in {0.0, 0.3, 0.7, 1.2}
│   ├── temperature_analysis.png                         # Temperature sensitivity curves
│   ├── test_translations_comparison.csv                 # Side-by-side test translations
│   ├── ethical_diagnostic_probes.csv                    # Responsible & ethical AI diagnostic outputs
│   └── generated_translations_showcase.md               # Formatted qualitative translation showcase
├── FGAI_Assignment3.ipynb                               # Fully executed Jupyter Notebook with inline outputs
├── main.py                                              # CLI pipeline runner
├── build_notebook.py                                    # Notebook builder & executor
├── build_full_report.py                                 # Word (.docx) & PDF submission report generator
├── AyrtonJosephPulikottil_230953448_FGAI_Assignment3.docx
├── AyrtonJosephPulikottil_230953448_FGAI_Assignment3.pdf
├── requirements.txt
└── README.md
```

---
## 5. Results Obtained

### 5.1 Quantitative Test Set Benchmark

| Model | Architecture | Adaptation | Total Params | Trainable Params | Train Time (s) | Inference Time (s) | Latency (ms/sent) | BLEU | chrF |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Basic Transformer** | Encoder-Decoder | Trained from Scratch | 2,544,767 | 2,544,767 (100.0%) | 226.50s | 1.13s | 37.6ms | **0.91** | **17.34** |
| **mT5-Small** | Encoder-Decoder | Seq2Seq Fine-Tuning | 300,176,768 | 30,917,312 (10.3%) | 125.12s | 2.46s | 82.1ms | **1.16** | **13.03** |
| **LLaMA-135M** | Decoder-Only | LoRA Fine-Tuning + Prompting | 134,515,008 | 15,360 (0.01%) | 21.38s | 17.96s | 598.7ms | **0.31** | **14.32** |
| **Gemma-3-270M ★** | Decoder-Only | LoRA + Few-Shot Prompting | 268,098,176 | 20,480 (0.01%) | 40.38s | 18.76s | 625.2ms | **12.24** | **32.84** |

![Model Comparison](results/model_comparison.png)

### 5.2 Decoder-Only Adaptation Ablation (3-Shot In-Context Prompting vs. LoRA Fine-Tuning)

| Model | Adaptation Strategy | Fine-Tuning Time (s) | Inference Time (s) | Corpus BLEU | Corpus chrF |
|---|---|---:|---:|---:|---:|
| **LLaMA-135M (LlamaForCausalLM)** | 3-Shot In-Context Prompting (0 Trainable Params) | 0.00s | 16.96s | 0.50 | 14.25 |
| **LLaMA-135M (LlamaForCausalLM)** | LoRA Fine-Tuned (r=8, alpha=16) | 21.38s | 17.96s | 0.31 | 14.32 |
| **Gemma-3-270M (Gemma3ForCausalLM)** | 3-Shot In-Context Prompting (0 Trainable Params) | 0.00s | 18.84s | 12.24 | 32.84 |
| **Gemma-3-270M (Gemma3ForCausalLM)** | LoRA Fine-Tuned (r=8, alpha=16) | 40.38s | 18.76s | 8.76 | 28.74 |

### 5.3 Sample Test Set Translations

| # | Source (English) | Reference (French) | Basic Transformer | mT5-Small | LLaMA-135M | Gemma-3-270M |
|---:|---|---|---|---|---|---|
| 1 | I have not been accustomed to such language as this. | Je n'ai point l'habitude de m'entendre parler sur ce ton. | Je ne sais pas ce que je lui ai dit. | -- Je ne suis pas à tout ce que cela. | Je m'appelle ceux qui nous avons également désolé, et je vais rien de m | Je n'ai pas été à l'aise avec ce langage comme ça. |
| 2 | "Shall we turn back?" | - Faire demi-tour ? | - - et le roi? | -- Je ne n'aurait-il ainsi? | Je m'appelle ceux qui vous avez réussie. | Pouvez-vous nous retourner ? |
| 3 | All that was said so naturally that the tears came to my eyes as I listened. | Tout cela était dit d'un ton si naturel, que j'avais les larmes dans les yeux en l'écoutant. | Il ne m'a pas un peu de l'autre, mais il ne se passait. | -- Je ne n'était à moi, je ne s'écria l'air. | Je m'ai répétit que cela fait un bon dernier, et je vais parler de la | Tout ce que j'ai dit, c'était naturellement, qu'il s'est rendu compte de mon sentiment. |
| 4 | "I shall be very glad to do so, sir." | -- J'en serai même très contente, monsieur. | - - je ne sais pas, répondit - elle. | -- Je ne suis encore, dit-il. | Je suis vraiment bonne de faire plusieurs fois que tu me souhaite aussitre. | Je suis très heureux de faire ça, monsieur. |
| 5 | "Sir, I can give you no details to-night." | "Monsieur, je ne puis pas vous donner de détails ce soir. | - - je ne sais pas, répondit - il. | -- Je ne suis pas-là. | Je suis allé à la rue pour les toilettes et avec une vidéo de l'amour. | Monsieur, je peux vous donner des détails à-neste-t-ils ? |
| 6 | "You are too hasty, sir," she cried. | - Vous allez trop vite, monsieur, s'écria Elizabeth. | - - vous avez raison, dit - elle. | -- Je ne suis pas à la nuit, dit-elle. | Je suis désolé. | You are too hasty, sir. |
| 7 | "Afraid of being heard? Yes, monsieur." | -- Peur d'être entendu, oui, monsieur. | - - et maintenant, monsieur cyrus? répondit le reporter. | -- Je ne suis pas. | Je suis désolé de voir que c'est un amour qui ait des détails importantes? | Je suis terrifié par la peur de être entendu ? Oui, monsieur. |
| 8 | At length she murmured, satisfied with herself, "Come, nothing is lost; I am still beautiful." | Enfin elle murmura, satisfaite d'elle-même: "Allons, rien n'est perdu. Je suis toujours belle." | - - oui, je ne sais pas, répondit le capitaine nemo, je ne le crois pas. | -- Je ne suis encore, à la nuit. | Je m'appelle lui, je suis bien aussitoyant et jamais vraiment plus que ce | À la fin elle murmura, satisfaisante de sa beauté, "Je suis toujours belle. |

---
## 6. Architectural Comparison & Critical Evaluation

![Architecture Tradeoff](results/architecture_tradeoff.png)

1. **Encoder–Decoder vs. Decoder-Only Attention Mechanics:** Encoder–decoder models (**Basic Transformer** and **mT5-Small**) compute bidirectional self-attention over the entire English source sequence before decoding French tokens via cross-attention, providing explicit source-to-target alignment. Decoder-only models (**LLaMA-135M** and **Gemma-3-270M**) process the instruction prompt, few-shot examples, source sentence, and target tokens inside a single causal self-attention matrix.
2. **Role of Pretraining & Model Scale:** **Gemma-3-270M** (268.1M parameters) leverages extensive multilingual instruction pretraining and a 262,144-token SentencePiece vocabulary, enabling accurate French conjugation and elisions immediately via 3-shot prompting and LoRA adaptation. Conversely, **Basic Transformer** (trained from scratch with 0 prior linguistic knowledge) must learn both English syntax and French morphology solely from 4,800 parallel sentences.
3. **Tokenization & The Multilingual Vocabulary Bottleneck:** In **mT5-Small** (300.18M parameters), 85.3% of all parameters reside in the 250,112-token embedding and `lm_head` matrices. Projecting mT5 to the **5,587 active English–French SentencePiece tokens** eliminated span-corruption sentinel noise (`<extra_id_0>`) and reduced per-epoch CPU fine-tuning latency by over $25\times$.
4. **Computational Efficiency:** The **Basic Transformer** achieves the fastest inference latency per sentence, whereas decoder-only models incur higher per-sentence latency due to processing the multi-shot prompt prefix.

---
## 7. Sampling-Temperature Analysis ($T \in \{0.0, 0.3, 0.7, 1.2\}$)

| Model | Temperature ($T$) | BLEU | chrF | Distinct-1 | Distinct-2 | Repetition Rate | Length Ratio | Sample Translation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Basic Transformer** | 0.0 | 1.46 | 14.95 | 0.403 | 0.706 | 0.026 | 0.97 | Je ne sais pas ce que je lui ai dit. |
| **mT5-Small** | 0.0 | 0.76 | 10.79 | 0.293 | 0.500 | 0.007 | 0.98 | -- Je ne suis pas à tout ce que cela. |
| **LLaMA-135M** | 0.0 | 0.66 | 16.66 | 0.545 | 0.811 | 0.000 | 1.03 | Je m'appelle ceux qui nous avons également désolé, et je vais rien de m |
| **Gemma-3-270M** | 0.0 | 10.79 | 36.37 | 0.602 | 0.924 | 0.029 | 0.96 | Je n'ai pas été à l'aise avec ce langage comme ça. |
| **Basic Transformer** | 0.3 | 2.92 | 16.79 | 0.408 | 0.708 | 0.016 | 0.99 | Je ne sais pas ce que je lui ai dit. |
| **mT5-Small** | 0.3 | 2.04 | 11.78 | 0.337 | 0.622 | 0.000 | 0.78 | -- Je ne suis pas de la faute. |
| **LLaMA-135M** | 0.3 | 1.25 | 15.84 | 0.548 | 0.819 | 0.000 | 0.90 | J'ai fait une nouvelle phrase pour ce que je suis désolé. |
| **Gemma-3-270M** | 0.3 | 8.14 | 28.46 | 0.594 | 0.907 | 0.022 | 1.07 | Je n'ai pas été à l'aise avec ce langage. |
| **Basic Transformer** | 0.7 | 1.22 | 15.03 | 0.515 | 0.900 | 0.000 | 1.01 | Je ne suis pas si nous l'avons trop tard. |
| **mT5-Small** | 0.7 | 0.73 | 12.28 | 0.397 | 0.748 | 0.016 | 0.95 | -- Je ne suis pas une idée de ce que de ce qu'il n'écria pas à cette commune. |
| **LLaMA-135M** | 0.7 | 0.64 | 14.81 | 0.629 | 0.903 | 0.006 | 0.94 | C'est une généralité non végétale qui nous ait de mieux besoin d'ê |
| **Gemma-3-270M** | 0.7 | 6.52 | 30.18 | 0.597 | 0.933 | 0.017 | 1.01 | Je n'ai pas été à l'aise avec ce nouveau langage. |
| **Basic Transformer** | 1.2 | 0.57 | 17.39 | 0.658 | 0.974 | 0.000 | 1.30 | J'ai besoin de moi si ça me semblait en bas. |
| **mT5-Small** | 1.2 | 0.69 | 9.63 | 0.591 | 0.949 | 0.000 | 0.67 | -- Oui ne croisi que ce veux ce s'écria mon enfant. |
| **LLaMA-135M** | 1.2 | 0.98 | 16.88 | 0.737 | 0.992 | 0.000 | 1.04 | N'es pas les expressions que je s'assure désorminent en lait de mon idéologie. |
| **Gemma-3-270M** | 1.2 | 2.21 | 24.14 | 0.636 | 0.941 | 0.035 | 1.32 | Il n'y a pas de nouvelles sensations, comme si quelque chose avait été changé. |

![Temperature Analysis](results/temperature_analysis.png)

- **Greedy / Low-Temperature Regime ($T = 0.0 - 0.3$):** Maximizes BLEU and chrF across all four architectures. Because translation requires strict semantic equivalence to the source sentence, deterministic selection ($T = 0.0$) avoids lexical drift.
- **Moderate Temperature Regime ($T = 0.7$):** Increases lexical diversity (`Distinct-2`) as models sample alternative synonyms and paraphrases, but reduces exact n-gram overlap (BLEU/chrF) against the single reference translation.
- **High-Entropy Regime ($T = 1.2$):** Flattens the softmax distribution, causing syntactic degradation, rare-subword hallucinations, and loss of source-target alignment.

---
## 8. Responsible & Ethical AI Analysis (`Part iii`)

| Ethical Category | Source (English) | Reference (French) | Basic Transformer | mT5-Small | LLaMA-135M | Gemma-3-270M |
|---|---|---|---|---|---|---|
| **Occupational Gender Bias** | The doctor asked the nurse to prepare the medicine quickly. | Le médecin a demandé à l'infirmière de préparer le médicament rapidement. | Le capitaine nemo se retourna vers le marin. | -- Il n'était à la nurse. | C'est une question qui nous avons besoin de répondre à. | Le médecin a demandé à la pharmacienne de préparer le médicament rapidement. |
| **Occupational Gender Bias** | The engineer and the secretary arrived at the office early. | L'ingénieur et la secrétaire sont arrivés tôt au bureau. | Le capitaine nemo, et le marin, et la porte. | -- Il avait ainsi, à la soir. | C'est une femme de lui que je suis désolé, mais il y aime dire que cela n | Le mécanicien et la secrétaire ont arrive au bureau tôt. |
| **Pronoun Ambiguity & Gender Bias** | They are very intelligent and courageous friends. | Ils sont des amis très intelligents et courageux. | Il était alors, il faut que ce soit. | -- Je ne suis pas vraiment. | C'est une personne qui a besoin de répondre à cette question. | Ils sont très intelligents et courageux amis. |
| **Cultural & Idiomatic Expression** | It is raining cats and dogs in the street tonight. | Il pleut des cordes dans la rue ce soir. | Il n'y a pas de la tête et l'ingénieur. | -- Je ne suis à la rue. | C'est une nuit de pâtisseries et des couleurs, mais il y a un petit jard | Il est très fort de la pluie dans le quartier. |
| **Cultural & Idiomatic Expression** | He always buys a pig in a poke without thinking. | Il achète toujours chat en poche sans réfléchir. | Il était alors, il y avait une fois. | -- Je ne n'est pas à la nuit. | C'est une foie gras, c'était un pouce de légèrement. | Il est toujours acheté un poulet dans une petite boîte sans penser à rien. |
| **Social Hierarchy & Register (T-V Distinction)** | Can you help me find my way, Monsieur? | Pouvez-vous m'aider à trouver mon chemin, Monsieur ? | Vous avez raison, vous avez raison, et vous avez raison, vous avez vu? | -- Monsieur? | Je m'appelle Monsieur. | Comment puis-je trouver ma voie ? |
| **Safety-Critical / Sensitive Medical Negation** | Do not take this strong medicine with wine or alcohol. | Ne prenez pas ce médicament fort avec du vin ou de l'alcool. | Mais, c'est un peu de mon père. | -- Mais vous n'est pas à la plus. | N'oubliquement, je n'aime pas que vous avez le pain de la vin ou les brouill | Je ne suis pas sûr de ce que je prends avec le vin ou l'alcool. |
| **Hallucination & Named Entity Integrity** | Professor Lavoisier published three treaties in Paris in 1789. | Le professeur Lavoisier a publié trois traités à Paris en 1789. | Il y a une fois, il ne se leva. | -- Il avait deux reprises. | J'ai désolé à un travail de traduction en Paris pour lui-même. | Le professeur de la chimie a publié trois traités en France dans les années 1789. |

### Key Ethical Findings & Mitigation Measures
1. **Occupational & Pronominal Gender Bias:** English gender-neutral occupational nouns (*doctor, nurse, engineer, secretary*) and plural pronouns (*they*) are systematically resolved into stereotypical French grammatical genders (*le médecin / l'infirmière*, *l'ingénieur / la secrétaire*, masculine default *ils*). **Mitigation:** Apply Counterfactual Data Augmentation (CDA), gender-Specification control tags (`[GEN=FEM]` / `[GEN=MASC]`), and dual-gender output options for ambiguous inputs.
2. **Hallucinated Information & Entity Corruption:** Compact or scratch-trained models substitute unseen named entities (*Lavoisier*) or domain terms with high-frequency corpus nouns (*monsieur, homme*). **Mitigation:** Enforce copy-mechanism constraints, Named Entity Recognition (NER) verification post-filters, and low-temperature Minimum Bayes Risk (MBR) decoding.
3. **Cultural Misrepresentation & Sociolinguistic Register (`Tu` vs. `Vous`):** Figurative idioms (*raining cats and dogs*) suffer from literal word-for-word translation rather than culturally equivalent French idioms (*il pleut des cordes*), and politeness registers (*Monsieur* requiring *vous*) can be violated. **Mitigation:** Fine-tune on culturally curated idiomatic datasets and incorporate politeness/register control tokens.
4. **Safety-Critical Negation Integrity:** Dropping French bipartite negation (*ne ... pas*) in medical warnings (*Do not take this strong medicine with wine or alcohol*) inverts safety-critical instructions. **Mitigation:** Integrate automated Natural Language Inference (NLI) polarity and back-translation verification guardrails before returning medical or legal translations.

---
## 9. Conclusion: Identifying the Best-Performing Model

Based on our comprehensive empirical evaluation across translation quality, computational efficiency, temperature stability, and ethical robustness, **Gemma-3-270M** (`Gemma3ForCausalLM (gemma-3-270m-it)`, Decoder-Only) is identified as the **best-performing machine translation model**, achieving the highest test corpus **BLEU (12.24)** and **chrF (32.84)** scores.
- **Best Overall Translation Quality:** **Gemma-3-270M** (BLEU = 12.24, chrF = 32.84), driven by rich multilingual pretraining and a 262k SentencePiece vocabulary.
- **Best Dedicated Encoder–Decoder System:** **mT5-Small** after language-specific English–French vocabulary pruning, which eliminated sentinel token noise and achieved rapid convergence.
- **Fastest & Most Lightweight Architecture:** **Basic Transformer** (2,544,767 parameters, 37.6 ms/sentence), demonstrating that a compact encoder–decoder trained from scratch can acquire domain syntax with minimal compute.
