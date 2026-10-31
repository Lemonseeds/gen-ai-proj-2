"""
Generates the submission Word document and PDF for Step 3 & 4 of the assignment:
YourName_RegNum_FGAI_Assignment3.docx / .pdf

Course: Foundations of Generative AI (ICT4403)
"""
import argparse
import os
import subprocess
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT


def build_submission_doc(
    student_name: str = "Ayrton Joseph Pulikottil",
    reg_number: str = "230953448",
    github_repo_url: str = "https://github.com/Lemonseeds/gen-ai-proj-2",
    output_filename: str | None = None,
):
    doc = Document()

    # Page Margins (1 inch all around)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Document Header
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_course = p_title.add_run("FOUNDATIONS OF GENERATIVE AI (ICT4403)\n")
    run_course.font.name = "Calibri"
    run_course.font.size = Pt(13)
    run_course.font.bold = True
    run_course.font.color.rgb = RGBColor(38, 70, 83)

    run_sub = p_title.add_run("Assignment 3: Neural Machine Translation Across Transformer Architectures\nSubmission Cover Sheet\n")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(17)
    run_sub.font.bold = True
    run_sub.font.color.rgb = RGBColor(42, 157, 143)

    p_div = doc.add_paragraph()
    p_div.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_div.paragraph_format.space_after = Pt(14)
    run_line = p_div.add_run("―" * 48)
    run_line.font.color.rgb = RGBColor(180, 180, 180)

    # Submission Details Table
    details = [
        ("Student Name", student_name),
        ("Registration Number", reg_number),
        ("Assignment Title", "ICT4403: Foundations of Generative AI — Assignment 3"),
        (
            "Title of Developed Autoregressive System",
            "Comparative English-to-French Neural Machine Translation & Ethical Auditing System (Basic Transformer, mT5, LLaMA, Gemma)",
        ),
        (
            "Selected Language Pair",
            "Source: English (en)  -->  Target: French (fr)",
        ),
        (
            "Name of Dataset Used",
            "OPUS Books Parallel Corpus (Helsinki-NLP/opus_books, en-fr)",
        ),
        (
            "Dataset Source Link",
            "https://huggingface.co/datasets/Helsinki-NLP/opus_books\n(OPUS Portal: https://opus.nlpl.eu/Books.php)",
        ),
        ("GitHub Repository Link", github_repo_url),
        (
            "Primary Deliverables in Repository",
            "• FGAI_Assignment3.ipynb (Fully executed master notebook with inline outputs & plots)\n"
            "• main.py & modular Python package in src/ (data, models, train_and_eval)\n"
            "• Saved PyTorch model checkpoint in models/basic_transformer.pt\n"
            "• Numerical metrics (CSV/JSON), test set translation samples, and 5 analytical figures in results/\n"
            "• Comprehensive README.md with mathematical formulations, evaluation tables, and ethical audit\n"
            "• Submission PDF & Word reports: AyrtonJosephPulikottil_230953448_FGAI_Assignment3.pdf",
        ),
    ]

    table = doc.add_table(rows=len(details), cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    for i, (label, val) in enumerate(details):
        row = table.rows[i]

        # Label cell
        cell_lbl = row.cells[0]
        cell_lbl.width = Inches(2.2)
        p_lbl = cell_lbl.paragraphs[0]
        p_lbl.paragraph_format.space_before = Pt(4)
        p_lbl.paragraph_format.space_after = Pt(4)
        run_l = p_lbl.add_run(label)
        run_l.font.name = "Calibri"
        run_l.font.bold = True
        run_l.font.size = Pt(10)
        run_l.font.color.rgb = RGBColor(38, 70, 83)

        # Value cell
        cell_val = row.cells[1]
        cell_val.width = Inches(4.3)
        p_val = cell_val.paragraphs[0]
        p_val.paragraph_format.space_before = Pt(4)
        p_val.paragraph_format.space_after = Pt(4)
        run_v = p_val.add_run(val)
        run_v.font.name = "Calibri"
        run_v.font.size = Pt(10)
        if "http" in val:
            run_v.font.color.rgb = RGBColor(33, 102, 172)
            run_v.font.underline = True

    # Executive Summary of Developed System & Experimental Findings
    p_sec = doc.add_paragraph()
    p_sec.paragraph_format.space_before = Pt(16)
    p_sec.paragraph_format.space_after = Pt(4)
    r_sec = p_sec.add_run("Executive Summary of Empirical Findings & Architectural Comparison")
    r_sec.font.name = "Calibri"
    r_sec.font.bold = True
    r_sec.font.size = Pt(11.5)
    r_sec.font.color.rgb = RGBColor(38, 70, 83)

    summary_text = (
        "This project implements an end-to-end comparative Neural Machine Translation (NMT) and ethical auditing system "
        "evaluating four distinct Transformer models across encoder–decoder and decoder-only paradigms on the publicly "
        "available OPUS Books parallel literature corpus (en -> fr):\n\n"
        "1. Basic Transformer (Encoder–Decoder, Trained from Scratch): Custom 6-layer PyTorch Seq2Seq model (d=192, H=6) "
        "trained for 8 epochs on CPU. Achieved test BLEU = 0.91, chrF = 17.34, with the lowest per-sentence inference latency "
        "(37.6 ms/sentence) and smallest memory footprint (2.54M parameters).\n\n"
        "2. mT5-Small (google/mt5-small, Encoder–Decoder, Pretrained & Fine-Tuned): Multilingual T5 (300.18M parameters) "
        "adapted via French/English sub-vocabulary projection (slicing 250,112 SentencePiece tokens down to 5,587 active subwords) "
        "and seq2seq fine-tuning. Achieved test BLEU = 1.16, chrF = 13.03, latency = 82.1 ms/sentence.\n\n"
        "3. LLaMA-135M (LlamaForCausalLM / SmolLM2-135M, Decoder-Only): Adapted via 3-Shot In-Context Prompting and top-layer "
        "LoRA fine-tuning (r=8, alpha=16). Achieved BLEU = 0.50, chrF = 14.25 (3-shot prompting) and chrF = 14.32 (LoRA fine-tuned), "
        "latency = 598.7 ms/sentence.\n\n"
        "4. Gemma-3-270M (Gemma3ForCausalLM / gemma-3-270m-it, Decoder-Only): Adapted via 3-Shot In-Context Prompting and LoRA fine-tuning. "
        "Achieved the highest translation fidelity across all benchmarks: BLEU = 12.24, chrF = 32.84, latency = 625.2 ms/sentence.\n\n"
        "Sampling-Temperature Dynamics:\n"
        "Testing across T in {0.0, 0.3, 0.7, 1.2} demonstrated that deterministic greedy decoding (T=0.0) maximizes lexical and morphological "
        "fidelity for NMT (Gemma chrF=36.4 at T=0.0 vs. 24.1 at T=1.2), while elevated temperature (T >= 0.7) increases lexical diversity (Distinct-2 > 0.90) "
        "at the expense of semantic drift and syntactic hallucination.\n\n"
        "Responsible & Ethical AI Auditing:\n"
        "Diagnostic probing across 8 targeted linguistic pairs revealed systemic occupational gender stereotyping (defaulting 'doctor'/'engineer' to masculine "
        "and 'nurse'/'secretary' to feminine), pronominal bias ('they' -> 'ils'), and risk of safety-critical negation inversion in medical prompts. "
        "Counterfactual data augmentation, formality-controlled prompting, and constrained decoding are proposed and analyzed as mitigation strategies."
    )

    p_sum = doc.add_paragraph(summary_text)
    p_sum.paragraph_format.space_after = Pt(10)
    p_sum.paragraph_format.line_spacing = 1.12
    for r in p_sum.runs:
        r.font.name = "Calibri"
        r.font.size = Pt(9.5)

    if not output_filename:
        safe_name = student_name.replace(" ", "")
        safe_reg = reg_number.replace(" ", "")
        output_filename = f"{safe_name}_{safe_reg}_FGAI_Assignment3.docx"

    doc.save(output_filename)
    print(f"Generated Word document: {output_filename}")
    return output_filename


def convert_docx_to_pdf_via_word(docx_path: str, pdf_path: str | None = None):
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
        Write-Output "PDF saved to {abs_pdf}"
    }} finally {{
        $word.Quit()
    }}
    """
    res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"Successfully converted to PDF: {abs_pdf}")
        return abs_pdf
    else:
        print(f"Error during PDF conversion: {res.stderr}")
        return None


if __name__ == "__main__":
    repo_url = "https://github.com/Lemonseeds/gen-ai-proj-2"
    
    # Generate Ayrton_REG_NUM_FGAI_Assignment3
    d1 = build_submission_doc("Ayrton", "230953448", repo_url, "Ayrton_REG_NUM_FGAI_Assignment3.docx")
    convert_docx_to_pdf_via_word(d1)

    print("Cover sheet generation complete!")
