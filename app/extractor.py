import pymupdf as fitz  # newer import for PyMuPDF
import os
import json
from tqdm import tqdm

def is_heading(text, font_size, heading_sizes):
    if font_size >= heading_sizes['H1']:
        return 'H1'
    elif font_size >= heading_sizes['H2']:
        return 'H2'
    elif font_size >= heading_sizes['H3']:
        return 'H3'
    return None

def extract_outline(pdf_path):
    doc = fitz.open(pdf_path)
    outline = []
    title = ""

    font_sizes = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_sizes.append(span["size"])
    font_sizes = sorted(set(font_sizes), reverse=True)

    heading_sizes = {
        "H1": font_sizes[0] if len(font_sizes) > 0 else 20,
        "H2": font_sizes[1] if len(font_sizes) > 1 else 17,
        "H3": font_sizes[2] if len(font_sizes) > 2 else 14,
    }

    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                line_text = ""
                line_size = 0
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                    line_text += " " + text if line_text else text
                    line_size = max(line_size, span["size"])
                if len(line_text) < 3:
                    continue
                heading_level = is_heading(line_text, line_size, heading_sizes)
                if heading_level:
                    if not title:
                        title = line_text
                    outline.append({
                        "level": heading_level,
                        "text": line_text,
                        "page": page_num
                    })

    return {
        "title": title,
        "outline": outline
    }

def process_all_pdfs(input_dir, output_dir):
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]

    for filename in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        pdf_path = os.path.join(input_dir, filename)
        result = extract_outline(pdf_path)
        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    if os.path.exists("/app/input"):
    # Running inside Docker
        input_dir = "/app/input"
        output_dir = "/app/output"
    else:
    # Running locally
        input_dir = "app/input"
        output_dir = "app/output"
    process_all_pdfs(input_dir, output_dir)