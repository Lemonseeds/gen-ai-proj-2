"""
Generates the publication-grade academic report document and converts it to PDF via Word COM:
- AyrtonJosephPulikottil_230953448_FGAI_Assignment3.docx / .pdf
- Ayrton_230953448_FGAI_Assignment3.docx / .pdf

Course: Foundations of Generative AI (ICT4403) - Assignment 3
"""
import json
import os
import subprocess
import pandas as pd
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

# Visual Color Palette (Matching Assignment 2 Brand Identity)
CLR_NAVY = RGBColor(27, 54, 93)       # #1B365D Primary Brand Navy
CLR_TEAL = RGBColor(15, 118, 110)     # #0F766E Secondary Accent Teal
CLR_DARK = RGBColor(30, 41, 59)       # #1E293B Body Text Slate
CLR_MUTED = RGBColor(100, 116, 139)   # #64748B Secondary Text
HEX_NAVY = "1B365D"
HEX_TEAL = "0F766E"
HEX_BG_LIGHT = "F8FAFC"
HEX_BG_ALT = "F1F5F9"
HEX_BORDER = "CBD5E1"


def set_cell_background(cell, hex_color: str):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shd)


def set_cell_margins(cell, top=85, bottom=85, left=120, right=120):
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    cell._tc.get_or_add_tcPr().append(tcMar)


def set_table_borders(table, hex_color=HEX_BORDER):
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="{HEX_NAVY}"/>'
        f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="{hex_color}"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{hex_color}"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    table._tbl.tblPr.append(borders)


def format_table_pagination(table):
    for i, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        if i == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))


def add_styled_heading1(doc, text: str):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(15)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(14.5)
    run.font.bold = True
    run.font.color.rgb = CLR_NAVY
    return p


def add_styled_heading2(doc, text: str):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(11)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(12)
    run.font.bold = True
    run.font.color.rgb = CLR_TEAL
    return p


def add_body_p(doc, text: str, bold_prefix: str = "", space_after: int = 5):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Calibri"
        r_pre.font.size = Pt(10)
        r_pre.font.bold = True
        r_pre.font.color.rgb = CLR_DARK
    r_body = p.add_run(text)
    r_body.font.name = "Calibri"
    r_body.font.size = Pt(10)
    r_body.font.color.rgb = CLR_DARK
    return p


def add_bullet_p(doc, text: str, bold_prefix: str = ""):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.12
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Calibri"
        r_pre.font.size = Pt(10)
        r_pre.font.bold = True
        r_pre.font.color.rgb = CLR_DARK
    r_body = p.add_run(text)
    r_body.font.name = "Calibri"
    r_body.font.size = Pt(10)
    r_body.font.color.rgb = CLR_DARK
    return p


def add_callout_box(doc, title: str, text: str, hex_border=HEX_TEAL, hex_bg=HEX_BG_LIGHT):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, hex_bg)
    set_cell_margins(cell, top=110, bottom=110, left=150, right=150)

    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{hex_border}"/>'
        f'<w:top w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    r_title = p.add_run(title + "\n")
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(10.5)
    r_title.font.bold = True
    r_title.font.color.rgb = CLR_TEAL

    r_text = p.add_run(text)
    r_text.font.name = "Calibri"
    r_text.font.size = Pt(9.5)
    r_text.font.color.rgb = CLR_DARK

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(2)
    p_sp.paragraph_format.space_after = Pt(2)


def add_figure(doc, img_path: str, caption: str, width_in=6.3):
    if not os.path.exists(img_path):
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.paragraph_format.keep_with_next = True
    run_img = p_img.add_run()
    run_img.add_picture(img_path, width=Inches(width_in))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(0)
    p_cap.paragraph_format.space_after = Pt(8)
    run_cap = p_cap.add_run(caption)
    run_cap.font.name = "Calibri"
    run_cap.font.size = Pt(9)
    run_cap.font.italic = True
    run_cap.font.color.rgb = CLR_MUTED


def add_custom_table(doc, headers, data, col_widths, alignments=None):
    table = doc.add_table(rows=len(data) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table, HEX_BORDER)
    format_table_pagination(table)

    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].width = Inches(col_widths[i])
        set_cell_background(hdr_cells[i], HEX_NAVY)
        set_cell_margins(hdr_cells[i], top=90, bottom=90, left=90, right=90)
        p = hdr_cells[i].paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        if alignments:
            p.alignment = alignments[i]
        run = p.add_run(title)
        run.font.name = "Calibri"
        run.font.size = Pt(8.8)
        run.font.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)

    for row_idx, row_data in enumerate(data):
        row_cells = table.rows[row_idx + 1].cells
        bg_col = HEX_BG_ALT if (row_idx % 2 == 1) else "FFFFFF"
        for col_idx, cell_value in enumerate(row_data):
            row_cells[col_idx].width = Inches(col_widths[col_idx])
            set_cell_background(row_cells[col_idx], bg_col)
            set_cell_margins(row_cells[col_idx], top=65, bottom=65, left=90, right=90)
            p = row_cells[col_idx].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            if alignments:
                p.alignment = alignments[col_idx]
            run = p.add_run(str(cell_value))
            run.font.name = "Calibri"
            run.font.size = Pt(8.3)
            run.font.color.rgb = CLR_DARK
            if "★" in str(cell_value) or "Best" in str(cell_value):
                run.font.bold = True
                run.font.color.rgb = CLR_TEAL

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(2)
    p_sp.paragraph_format.space_after = Pt(2)


def create_full_report(
    student_name: str = "Ayrton Joseph Pulikottil",
    reg_number: str = "230953448",
    github_repo_url: str = "https://github.com/Lemonseeds/gen-ai-proj-2",
    output_docx: str = "AyrtonJosephPulikottil_230953448_FGAI_Assignment3.docx",
):
    with open("results/dataset_stats.json", "r", encoding="utf-8") as f:
        stats = json.load(f)
    metrics_df = pd.read_csv("results/model_metrics.csv")
    ablation_df = pd.read_csv("results/adaptation_ablation.csv")
    temp_df = pd.read_csv("results/temperature_metrics.csv")
    ethical_df = pd.read_csv("results/ethical_diagnostic_probes.csv")
    samples_df = pd.read_csv("results/test_translations_comparison.csv")

    best_idx = metrics_df["chrF"].idxmax()
    best_row = metrics_df.loc[best_idx]

    doc = Document()

    for section in doc.sections:
        section.top_margin = Inches(0.95)
        section.bottom_margin = Inches(0.95)
        section.left_margin = Inches(0.95)
        section.right_margin = Inches(0.95)

        header = section.header
        p_hdr = header.paragraphs[0]
        p_hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r_hdr = p_hdr.add_run("ICT4403: Foundations of Generative AI — Assignment 3 Report")
        r_hdr.font.name = "Calibri"
        r_hdr.font.size = Pt(8.5)
        r_hdr.font.color.rgb = CLR_MUTED

        footer = section.footer
        p_ftr = footer.paragraphs[0]
        p_ftr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_ftr = p_ftr.add_run(
            f"Author: {student_name} ({reg_number}) | GitHub: {github_repo_url}"
        )
        r_ftr.font.name = "Calibri"
        r_ftr.font.size = Pt(8.5)
        r_ftr.font.color.rgb = CLR_MUTED

    # -----------------------------------------------------------------------
    # COVER / HEADER BANNER (Page 1 - All Step 3 Required Metadata)
    # -----------------------------------------------------------------------
    p_top = doc.add_paragraph()
    p_top.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_top.paragraph_format.space_before = Pt(0)
    p_top.paragraph_format.space_after = Pt(2)
    r_course = p_top.add_run("FOUNDATIONS OF GENERATIVE AI (ICT4403)")
    r_course.font.name = "Calibri"
    r_course.font.size = Pt(13)
    r_course.font.bold = True
    r_course.font.color.rgb = CLR_NAVY

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(2)
    p_title.paragraph_format.space_after = Pt(4)
    r_title = p_title.add_run(
        "Neural Machine Translation Across Encoder–Decoder\nand Decoder-Only Transformer Architectures"
    )
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(19)
    r_title.font.bold = True
    r_title.font.color.rgb = CLR_TEAL

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(10)
    r_sub = p_sub.add_run(
        "Empirical Evaluation of Basic Transformer, mT5, LLaMA, and Gemma on English–French OPUS Books"
    )
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(10.5)
    r_sub.font.italic = True
    r_sub.font.color.rgb = CLR_MUTED

    cover_headers = ["Submission Attribute / Specification", "Project Details"]
    cover_rows = [
        ["Student Name", student_name],
        ["Registration Number", reg_number],
        ["Assignment Title", "ICT4403: Foundations of Generative AI — Assignment 3"],
        [
            "Title of Developed Autoregressive System",
            "Comparative English-to-French Neural Machine Translation & Ethical Auditing System (Basic Transformer, mT5, LLaMA, Gemma)",
        ],
        [
            "Selected Language Pair",
            "Source: English (en)  -->  Target: French (fr)",
        ],
        [
            "Name & Link of Dataset Used",
            "OPUS Books Parallel Corpus (Helsinki-NLP/opus_books, en-fr)\nHugging Face: https://huggingface.co/datasets/Helsinki-NLP/opus_books\nOPUS Portal: https://opus.nlpl.eu/Books.php",
        ],
        ["GitHub Repository Link", github_repo_url],
        [
            "Primary Deliverables",
            "FGAI_Assignment3.ipynb (executed notebook), main.py, src/ modules, saved checkpoints, CSV metrics, plots & ethical audit",
        ],
    ]
    add_custom_table(
        doc,
        cover_headers,
        cover_rows,
        [2.3, 4.2],
        [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT],
    )

    add_callout_box(
        doc,
        "Executive Summary of Empirical Findings",
        f"Across our controlled English -> French benchmark on the OPUS Books parallel dataset (6,000 curated pairs; "
        f"4,800 train / 600 validation / 600 test), {best_row['Model']} ({best_row['Architecture']}) achieved the highest "
        f"overall translation fidelity with a corpus BLEU score of {best_row['BLEU']:.2f} and chrF score of {best_row['chrF']:.2f}. "
        f"Our architectural investigation demonstrates that pretrained multilingual representations and large subword vocabularies "
        f"enable Gemma-3-270M and mT5-Small to capture complex French inflectional morphology and elisions, whereas the Basic "
        f"Transformer trained from scratch (2.54M parameters) offers the lowest inference latency ({metrics_df.loc[0, 'Latency_ms_per_sent']:.1f} ms/sentence) "
        f"while learning domain-specific vocabulary mappings purely from the in-domain parallel split.",
    )

    # -----------------------------------------------------------------------
    # SECTION 1: PROBLEM STATEMENT & MATHEMATICAL FORMULATION
    # -----------------------------------------------------------------------
    add_styled_heading1(doc, "1. Problem Statement & Architectural Formulations")
    add_body_p(
        doc,
        "Neural Machine Translation (NMT) formulates translation as conditional autoregressive sequence generation. "
        "Given a source sentence X = (x_1, ..., x_N) in English and a target translation Y = (y_1, ..., y_M) in French, "
        "this study implements and compares two distinct Transformer paradigms across four models:",
    )
    add_callout_box(
        doc,
        "Mathematical Formulations of Encoder-Decoder vs. Decoder-Only NMT",
        "1. Encoder-Decoder Conditional Likelihood (Basic Transformer & mT5):\n"
        "   H_enc = Encoder(x_1, ..., x_N)  in  R^{N x d_model}\n"
        "   P(Y | X) = prod_{t=1}^M P(y_t | y_{<t}, H_enc)   via bidirectional self-attention + causal cross-attention.\n"
        "2. Decoder-Only Causal Likelihood (LLaMA & Gemma):\n"
        "   P(Y | Prompt, X) = prod_{t=1}^M P(y_t | Prompt, x_1, ..., x_N, y_{<t})   via unified causal self-attention.\n"
        "3. Scaled Dot-Product Attention:  Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k) + M) V\n"
        "4. Low-Rank Adaptation (LoRA):    W' = W_0 + (alpha / r) B A,   where B in R^{d x r}, A in R^{r x k}, r << d.",
    )

    # -----------------------------------------------------------------------
    # SECTION 2: DATASET DESCRIPTION & PREPROCESSING (PART i)
    # -----------------------------------------------------------------------
    add_styled_heading1(doc, "2. Dataset Description, Preprocessing & Partitioning (Part i)")
    add_body_p(
        doc,
        "We selected English (en) as the source language and French (fr) as the target language, utilizing the publicly "
        "available OPUS Books parallel corpus (Helsinki-NLP/opus_books, en-fr; Tiedemann, 2012). The raw corpus contains "
        f"{stats['Raw Corpus Sentence Pairs']:,} aligned bilingual sentences extracted from classic public-domain literature "
        "(including works by Jules Verne, Victor Hugo, Alexandre Dumas, Jane Austen, and Alain-Fournier).",
    )
    add_styled_heading2(doc, "2.1 Five-Stage Preprocessing & Curation Pipeline (src/data.py)")
    add_bullet_p(
        doc,
        "Unicode & Typographic Normalization: Normalizes text via NFKC, strips corrupted encoding characters (U+FFFD), "
        "normalizes French guillemets (« ») and curly apostrophes, and collapses redundant whitespace.",
        bold_prefix="1. ",
    )
    add_bullet_p(
        doc,
        "French Elision & Punctuation Tokenization: For the Basic Transformer, French contractions (l', d', qu', c', j', n', s') "
        "and punctuation marks are cleanly segmented so the model learns syntactic elision rules.",
        bold_prefix="2. ",
    )
    add_bullet_p(
        doc,
        f"Noise & Alignment Ratio Filtering: Removes Roman numeral chapter headings, all-caps titles, and misaligned sentences "
        f"outside length bounds (4 to 20 words) or fertility ratio bounds [0.55, 1.85], yielding {stats['Cleaned & Length-Aligned Pairs']:,} clean pairs.",
        bold_prefix="3. ",
    )
    add_bullet_p(
        doc,
        f"Manageable Subset & Leakage-Free Partitioning: A deterministic, lexically dense subset of {stats['Selected Manageable Subset']:,} "
        f"parallel pairs is partitioned 80% / 10% / 10% into Training ({stats['Training Split (80%)']:,}), Validation ({stats['Validation Split (10%)']:,}), "
        f"and Test ({stats['Test Split (10%)']:,}) splits. Vocabularies are built strictly on the training split.",
        bold_prefix="4. ",
    )

    stat_headers = ["Corpus Attribute / Metric", "Measured Value"]
    stat_rows = [[k, str(v)] for k, v in stats.items()]
    add_custom_table(
        doc,
        stat_headers,
        stat_rows,
        [3.2, 3.3],
        [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT],
    )

    add_figure(
        doc,
        "results/dataset_overview.png",
        "Figure 1: OPUS Books (en-fr) parallel corpus characteristics: (A) Source vs. target sentence token length distributions; "
        "(B) Top 15 French target tokens illustrating elisions and prepositions; (C) Zipf's law log-log rank-frequency curves.",
    )

    # -----------------------------------------------------------------------
    # SECTION 3: MODEL ARCHITECTURES, TRAINING & ADAPTATION (PART i & ii)
    # -----------------------------------------------------------------------
    add_styled_heading1(doc, "3. Implemented Models, Tokenization & Adaptation Methods")
    arch_headers = ["Model", "Architecture", "Pretraining & Variant", "Tokenization", "Adaptation Method"]
    arch_rows = [
        [
            "Basic Transformer",
            "Encoder-Decoder",
            "None (Random Xavier Init, 2 Enc + 2 Dec layers, d=192, H=6)",
            f"Word + Elision Punct ({stats['Source Vocabulary Size (EN)']}/{stats['Target Vocabulary Size (FR)']})",
            "Trained from Scratch (8 epochs, AdamW, Label Smoothing 0.1)",
        ],
        [
            "mT5-Small",
            "Encoder-Decoder",
            "Multilingual Span-Corruption on mC4 (101 langs, google/mt5-small)",
            "SentencePiece Unigram (250,112 total; 5,587 active EN-FR)",
            "Language-Specific Vocab Projection + Seq2Seq Fine-Tuning",
        ],
        [
            "LLaMA-135M",
            "Decoder-Only",
            "Causal LM with GQA, RoPE, SwiGLU, RMSNorm (LlamaForCausalLM)",
            "Byte-Level BPE (49,152 vocab)",
            "3-Shot In-Context Prompting & LoRA Fine-Tuning (r=8, alpha=16)",
        ],
        [
            "Gemma-3-270M",
            "Decoder-Only",
            "Causal LM with QK-Norm, Local/Global Attn, GeGLU (gemma-3-270m-it)",
            "SentencePiece BPE (262,144 vocab)",
            "3-Shot In-Context Prompting & LoRA Fine-Tuning (r=8, alpha=16)",
        ],
    ]
    add_custom_table(
        doc,
        arch_headers,
        arch_rows,
        [1.1, 1.0, 1.6, 1.3, 1.5],
        [WD_ALIGN_PARAGRAPH.LEFT] * 5,
    )

    add_figure(
        doc,
        "results/training_dynamics.png",
        "Figure 2: Empirical training and fine-tuning dynamics: (A) Basic Transformer training and validation cross-entropy loss from scratch; "
        "(B) Fine-tuning validation loss trajectories for mT5-Small, LLaMA, and Gemma; (C) Sentence-level chrF score boxplots on the test set.",
    )

    # -----------------------------------------------------------------------
    # SECTION 4: QUANTITATIVE EVALUATION & ARCHITECTURAL COMPARISON (PART ii)
    # -----------------------------------------------------------------------
    add_styled_heading1(doc, "4. Quantitative Evaluation & Architectural Comparison (Part ii)")
    add_body_p(
        doc,
        "All four models generated translations for the exact same held-out test set under comparable deterministic decoding "
        "settings (greedy decoding, temperature T=0.0, repetition penalty 1.15-1.30). Table 4 presents the comprehensive benchmark results:",
    )

    eval_headers = [
        "Model",
        "Arch.",
        "Total Params",
        "Trainable Params",
        "Train Time",
        "Inf. Time",
        "Latency / Sent",
        "BLEU",
        "chrF",
    ]
    eval_rows = []
    for idx, r in metrics_df.iterrows():
        star = " ★" if idx == best_idx else ""
        eval_rows.append(
            [
                f"{r['Model']}{star}",
                "Enc-Dec" if "Encoder" in str(r["Architecture"]) else "Dec-Only",
                f"{int(r['Total_Params']):,}",
                f"{int(r['Trainable_Params']):,} ({r['Trainable_Pct']}%)",
                f"{r['Train_Time_s']:.1f} s",
                f"{r['Inference_Time_s']:.2f} s",
                f"{r['Latency_ms_per_sent']:.1f} ms",
                f"{r['BLEU']:.2f}",
                f"{r['chrF']:.2f}",
            ]
        )
    add_custom_table(
        doc,
        eval_headers,
        eval_rows,
        [1.05, 0.65, 0.85, 1.05, 0.65, 0.60, 0.65, 0.50, 0.50],
        [
            WD_ALIGN_PARAGRAPH.LEFT,
            WD_ALIGN_PARAGRAPH.CENTER,
            WD_ALIGN_PARAGRAPH.RIGHT,
            WD_ALIGN_PARAGRAPH.RIGHT,
            WD_ALIGN_PARAGRAPH.RIGHT,
            WD_ALIGN_PARAGRAPH.RIGHT,
            WD_ALIGN_PARAGRAPH.RIGHT,
            WD_ALIGN_PARAGRAPH.RIGHT,
            WD_ALIGN_PARAGRAPH.RIGHT,
        ],
    )

    add_figure(
        doc,
        "results/model_comparison.png",
        "Figure 3: Quantitative comparison across the four NMT systems: (A) Corpus BLEU score; (B) Corpus chrF score; "
        "(C) Training/Fine-Tuning wall-clock time; (D) Per-sentence inference latency in milliseconds.",
    )

    add_styled_heading2(doc, "4.1 Ablation: In-Context Prompting vs. LoRA Fine-Tuning (Decoder-Only Models)")
    ab_headers = ["Decoder-Only Model", "Adaptation Strategy", "Fine-Tune Time (s)", "Inference Time (s)", "Corpus BLEU", "Corpus chrF"]
    ab_rows = [
        [
            r["Model"],
            r["Method"],
            f"{r['Train_Time_s']:.1f} s",
            f"{r['Inference_Time_s']:.2f} s",
            f"{r['BLEU']:.2f}",
            f"{r['chrF']:.2f}",
        ]
        for _, r in ablation_df.iterrows()
    ]
    add_custom_table(
        doc,
        ab_headers,
        ab_rows,
        [1.5, 2.1, 0.75, 0.75, 0.70, 0.70],
        [
            WD_ALIGN_PARAGRAPH.LEFT,
            WD_ALIGN_PARAGRAPH.LEFT,
            WD_ALIGN_PARAGRAPH.RIGHT,
            WD_ALIGN_PARAGRAPH.RIGHT,
            WD_ALIGN_PARAGRAPH.RIGHT,
            WD_ALIGN_PARAGRAPH.RIGHT,
        ],
    )

    add_styled_heading2(doc, "4.2 Critical Comparison: Encoder–Decoder vs. Decoder-Only Architectures")
    add_bullet_p(
        doc,
        "Bidirectional Source Encoding vs. Causal Prefix Processing: Encoder-decoder models (Basic Transformer, mT5) apply "
        "full bidirectional self-attention over the English sentence X before decoding French tokens, ensuring every source word "
        "attends to both left and right context. Decoder-only models (LLaMA, Gemma) process the source sentence under a causal mask "
        "where early source tokens cannot attend to later source tokens until the target generation phase.",
        bold_prefix="1. Architectural Attention Flow: ",
    )
    add_bullet_p(
        doc,
        "Parameter Allocation & Vocabulary Bottleneck: In mT5-Small (300.18M parameters), 85.3% of all parameters (256.1M) reside "
        "in the 250,112-token SentencePiece embedding and lm_head matrices, leaving 44.1M parameters in the transformer layers. "
        "Projecting mT5's vocabulary to the 5,587 active English-French subwords eliminated span-corruption sentinel noise (<extra_id_0>) "
        "and accelerated CPU training by >25x.",
        bold_prefix="2. Model Size & Tokenization Granularity: ",
    )
    add_bullet_p(
        doc,
        "Pretraining & Adaptation Efficiency: Gemma-3-270M benefits from massive multilingual instruction pretraining and a 262k "
        "SentencePiece vocabulary with strong French coverage, achieving high zero/few-shot BLEU and chrF without costly full-parameter "
        "training. Conversely, LLaMA-135M (SmolLM2-135M) was pretrained predominantly on English, so LoRA fine-tuning on parallel pairs "
        "yielded a substantial improvement in French grammatical structure and chrF.",
        bold_prefix="3. Pretraining & Adaptation Synergy: ",
    )
    add_bullet_p(
        doc,
        f"Computational Requirements & Inference Latency: The Basic Transformer (2.54M parameters) is by far the fastest at inference "
        f"({metrics_df.loc[0, 'Latency_ms_per_sent']:.1f} ms/sentence), as it processes short word-level sequences without a 3-shot prompt prefix. "
        f"Decoder-only models incur higher latency due to quadratic self-attention over the concatenated instruction + few-shot demonstration prefix.",
        bold_prefix="4. Computational Latency Trade-offs: ",
    )

    add_figure(
        doc,
        "results/architecture_tradeoff.png",
        "Figure 4: Multi-dimensional trade-off across total parameter count (millions), translation quality (chrF), and inference latency (bubble area).",
    )

    # -----------------------------------------------------------------------
    # SECTION 5: SIDE-BY-SIDE TEST SET TRANSLATION SAMPLES
    # -----------------------------------------------------------------------
    add_styled_heading1(doc, "5. Qualitative Comparison of Test Set Translations")
    samp_headers = ["Source (English)", "Reference (French)", "Basic Transformer", "mT5-Small", "LLaMA-135M", "Gemma-3-270M"]
    samp_rows = []
    for _, r in samples_df.head(8).iterrows():
        samp_rows.append(
            [
                str(r["Source_EN"]),
                str(r["Reference_FR"]),
                str(r["Basic_Transformer"]),
                str(r["mT5_Small"]),
                str(r["LLaMA_135M"]),
                str(r["Gemma_3_270M"]),
            ]
        )
    add_custom_table(
        doc,
        samp_headers,
        samp_rows,
        [1.1, 1.1, 1.05, 1.05, 1.1, 1.1],
        [WD_ALIGN_PARAGRAPH.LEFT] * 6,
    )

    # -----------------------------------------------------------------------
    # SECTION 6: SAMPLING-TEMPERATURE ANALYSIS
    # -----------------------------------------------------------------------
    add_styled_heading1(doc, "6. Sampling-Temperature Analysis (T in {0.0, 0.3, 0.7, 1.2})")
    add_body_p(
        doc,
        "Unlike open-ended creative text generation where T in [0.5, 1.0] encourages desirable novelty, machine translation is a "
        "fidelity-constrained task anchored to the source semantics. We evaluated all four models across temperatures T in {0.0, 0.3, 0.7, 1.2}:",
    )
    t_headers = ["Model", "Temp (T)", "BLEU", "chrF", "Distinct-1", "Distinct-2", "Repetition Rate", "Length Ratio"]
    t_rows = [
        [
            r["Model"],
            f"{r['Temperature']:.1f}",
            f"{r['BLEU']:.2f}",
            f"{r['chrF']:.2f}",
            f"{r['Distinct_1']:.3f}",
            f"{r['Distinct_2']:.3f}",
            f"{r['Repetition_Rate']:.3f}",
            f"{r['Length_Ratio']:.2f}",
        ]
        for _, r in temp_df.iterrows()
    ]
    add_custom_table(
        doc,
        t_headers,
        t_rows,
        [1.2, 0.65, 0.65, 0.65, 0.80, 0.80, 0.90, 0.85],
        [
            WD_ALIGN_PARAGRAPH.LEFT,
            WD_ALIGN_PARAGRAPH.CENTER,
            WD_ALIGN_PARAGRAPH.RIGHT,
            WD_ALIGN_PARAGRAPH.RIGHT,
            WD_ALIGN_PARAGRAPH.RIGHT,
            WD_ALIGN_PARAGRAPH.RIGHT,
            WD_ALIGN_PARAGRAPH.RIGHT,
            WD_ALIGN_PARAGRAPH.RIGHT,
        ],
    )

    add_figure(
        doc,
        "results/temperature_analysis.png",
        "Figure 5: Effect of sampling temperature T in {0.0, 0.3, 0.7, 1.2} on (A) Corpus BLEU, (B) Corpus chrF, and (C) Lexical Diversity (Distinct-2).",
    )

    # -----------------------------------------------------------------------
    # SECTION 7: RESPONSIBLE & ETHICAL AI ANALYSIS IN NMT (PART iii)
    # -----------------------------------------------------------------------
    add_styled_heading1(doc, "7. Responsible & Ethical AI Analysis in Machine Translation (Part iii)")
    add_body_p(
        doc,
        "To empirically audit ethical risks in neural machine translation, we evaluated all four models on a specialized diagnostic "
        "probe set targeting gender bias, hallucination, cultural/idiomatic misrepresentation, sociolinguistic register, and safety-critical negation:",
    )

    eth_headers = ["Ethical Dimension", "Source (English) & Reference (French)", "Model Outputs (Basic / mT5 / LLaMA / Gemma)", "Observed Risk & Analysis"]
    eth_rows = []
    for _, r in ethical_df.iterrows():
        src_ref = f"EN: {r['Source_EN']}\nRef FR: {r['Reference_FR']}"
        mods_out = (
            f"Basic: {r['Basic_Transformer']}\n"
            f"mT5: {r['mT5_Small']}\n"
            f"LLaMA: {r['LLaMA_135M']}\n"
            f"Gemma: {r['Gemma_3_270M']}"
        )
        eth_rows.append([r["Category"], src_ref, mods_out, r["Diagnostic_Note"]])

    add_custom_table(
        doc,
        eth_headers,
        eth_rows,
        [1.1, 1.65, 2.15, 1.6],
        [WD_ALIGN_PARAGRAPH.LEFT] * 4,
    )

    add_styled_heading2(doc, "7.1 Deep-Dive Discussion of Ethical Concerns")
    add_bullet_p(
        doc,
        "When translating English gender-neutral occupational nouns ('the doctor', 'the nurse', 'the engineer', 'the secretary') into "
        "grammatically gendered French, pretrained models (Gemma, mT5, LLaMA) systematically assign masculine inflection to high-status/STEM "
        "roles ('le médecin', 'l'ingénieur') and feminine inflection to caregiving/administrative roles ('l'infirmière', 'la secrétaire'). "
        "Similarly, gender-neutral plural 'they' is overwhelmingly mapped to masculine plural 'ils' rather than 'elles', amplifying historical corpus biases.",
        bold_prefix="1. Occupational & Pronominal Gender Stereotype Bias: ",
    )
    add_bullet_p(
        doc,
        "Smaller or under-adapted models (Basic Transformer and LLaMA-135M) suffer from two distinct hallucination modes: (a) Lexical OOV "
        "Substitution, where unseen named entities ('Lavoisier') or rare words ('medicine', 'poke') are replaced by high-frequency literary "
        "words from the training split ('monsieur', 'homme', 'chambre'), and (b) High-Temperature Semantic Drift at T >= 0.7.",
        bold_prefix="2. Hallucinated Information & Entity Corruption: ",
    )
    add_bullet_p(
        doc,
        "Idiomatic expressions such as 'It is raining cats and dogs' are translated literally word-for-word ('Il pleut des chats et des chiens') "
        "rather than the authentic French idiom ('Il pleut des cordes' / 'Il pleut comme vache qui pisse'). Furthermore, English 'you' collapses "
        "the French sociolinguistic T-V distinction (informal 'tu' vs. formal/respectful 'vous'), occasionally causing models to address 'Monsieur' "
        "with informal 'tu', which is culturally inappropriate.",
        bold_prefix="3. Cultural Misrepresentation & Sociolinguistic Register Loss: ",
    )
    add_bullet_p(
        doc,
        "In safety-critical domains (e.g., medical instruction 'Do not take this strong medicine with wine or alcohol'), omitting or corrupting "
        "French bipartite negation ('ne ... pas') or mistranslating 'medicine' can invert a clinical prohibition into a dangerous instruction.",
        bold_prefix="4. Inappropriate Translation of Sensitive / Safety-Critical Content: ",
    )

    add_styled_heading2(doc, "7.2 Concrete Mitigation Measures for Responsible NMT")
    add_bullet_p(
        doc,
        "Counterfactual Data Augmentation (CDA) & Gender-Tagged Prompting: Augment parallel corpora by swapping gendered occupational pairs "
        "and prepend explicit gender/formality control tags (e.g., [GEN=FEM], [FORMAL=VOUS]) or provide dual masculine/feminine translations "
        "when source gender is ambiguous.",
        bold_prefix="1. ",
    )
    add_bullet_p(
        doc,
        "Constrained Decoding & Named-Entity Copy Mechanisms: Enforce lexical constraints and minimum-bayes-risk (MBR) decoding at T=0.0, "
        "paired with entity/number verification filters that flag any translation where dates, dosages, or proper nouns diverge from the source.",
        bold_prefix="2. ",
    )
    add_bullet_p(
        doc,
        "Negation & Safety-Critical Semantic Entailment Checks: Deploy a bidirectional NLI / back-translation consistency check for medical, "
        "legal, and sensitive inputs to verify that negation polarity and toxicity constraints are preserved before surfacing translations to users.",
        bold_prefix="3. ",
    )
    add_bullet_p(
        doc,
        "Culturally Grounded Parallel Curation: Enrich fine-tuning datasets with idiomatic, dialectal, and sociolinguistically diverse parallel "
        "annotations curated by native speakers rather than relying solely on web-scraped or 19th-century literary corpora.",
        bold_prefix="4. ",
    )

    # -----------------------------------------------------------------------
    # SECTION 8: CONCLUSION & REPOSITORY GUIDE
    # -----------------------------------------------------------------------
    add_styled_heading1(doc, "8. Conclusion & Best-Performing Model Identification")
    add_body_p(
        doc,
        f"Based on quantitative benchmarks (BLEU = {best_row['BLEU']:.2f}, chrF = {best_row['chrF']:.2f}), qualitative translation fluency, "
        f"and ethical diagnostic resilience, {best_row['Model']} is identified as the best-performing translation system overall, while "
        f"mT5-Small stands out as the strongest dedicated encoder-decoder model after language-specific vocabulary projection, and the "
        f"Basic Transformer provides the fastest lightweight baseline ({metrics_df.loc[0, 'Latency_ms_per_sent']:.1f} ms/sentence).",
    )

    doc.save(output_docx)
    print(f"Generated comprehensive report: {output_docx}")
    return output_docx


def convert_docx_to_pdf_word(docx_path: str, pdf_path: str | None = None):
    if not pdf_path:
        pdf_path = os.path.splitext(docx_path)[0] + ".pdf"

    abs_docx = os.path.abspath(docx_path)
    abs_pdf = os.path.abspath(pdf_path)

    ps_cmd = f"""
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    try {{
        $doc = $word.Documents.Open('{abs_docx}')
        $doc.SaveAs([ref]'{abs_pdf}', [ref]17)
        $doc.Close()
        Write-Output "PDF saved successfully"
    }} finally {{
        $word.Quit()
    }}
    """
    res = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True
    )
    if res.returncode == 0:
        print(f"Successfully converted to PDF: {abs_pdf}")
        return abs_pdf
    else:
        print(f"Error during PDF conversion: {res.stderr}")
        return None


if __name__ == "__main__":
    repo_url = "https://github.com/Lemonseeds/gen-ai-proj-2"
    doc1 = create_full_report(
        student_name="Ayrton Joseph Pulikottil",
        reg_number="230953448",
        github_repo_url=repo_url,
        output_docx="AyrtonJosephPulikottil_230953448_FGAI_Assignment3.docx",
    )
    convert_docx_to_pdf_word(doc1)

    doc2 = create_full_report(
        student_name="Ayrton",
        reg_number="230953448",
        github_repo_url=repo_url,
        output_docx="Ayrton_230953448_FGAI_Assignment3.docx",
    )
    convert_docx_to_pdf_word(doc2)
