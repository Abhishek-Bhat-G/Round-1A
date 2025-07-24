import fitz
import os, json, re
import pandas as pd
from tqdm import tqdm
from app.utils import normalize_text, extract_features

PDF_DIR = "app/input"
GT_DIR = "groundtruth/"

def extract_spans(pdf_path):
    doc = fitz.open(pdf_path)
    spans = []
    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["text"].strip():
                        spans.append({
                            "text": span["text"],
                            "font": span.get("font", ""),
                            "size": span["size"],
                            "bbox": span["bbox"],
                            "page": page_num
                        })
    return spans

def generator():
    rows = []
    for file in tqdm(os.listdir(GT_DIR)):
        if not file.endswith(".json"):
            continue
        base = file.replace(".json", "")
        json_path = os.path.join(GT_DIR, file)
        pdf_path = os.path.join(PDF_DIR, base + ".pdf")

        if not os.path.exists(pdf_path):
            continue

        with open(json_path) as jf:
            gt = json.load(jf)

        title_text = normalize_text(gt.get("title", ""))
        gt_outline = {normalize_text(entry["text"]): entry["level"] for entry in gt.get("outline", [])}

        spans = extract_spans(pdf_path)
        for i, span in enumerate(spans):
            norm = normalize_text(span["text"])
            label = (
                "Title" if norm == title_text
                else gt_outline.get(norm, "None")
            )
            feat = extract_features(span, spans[i - 1] if i > 0 else None)
            feat["label"] = label
            rows.append(feat)

    df = pd.DataFrame(rows)
    df.to_csv("training_data.csv", index=False)
    print("✅ training_data.csv generated!")

if __name__ == "__main__":
    generator()
