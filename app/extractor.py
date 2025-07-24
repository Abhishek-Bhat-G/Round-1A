import os
import json
import fitz
import joblib
import re
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Optional

class PDFOutlineExtractor:
    def __init__(self):
        self.model = joblib.load("app/model.pkl")
        self.le = joblib.load("app/label_encoder.pkl")
        self.FEATURE_ORDER = [
            "font_size", "is_bold", "x0", "word_count", "capital_ratio",
            "ends_colon", "numbered", "y_distance", "is_centered",
            "starts_capital", "all_caps", "line_length", "page_position",
            "prev_font_size", "prev_is_bold"
        ]
        self.title_font_threshold = 0.9  # Relative to max font size
        self.min_heading_words = 2  # Minimum words to consider as heading

    def extract_spans(self, pdf) -> List[Dict]:
        """Extract all text spans with formatting and positional metadata"""
        spans = []
        for page_num, page in enumerate(pdf, 1):
            page_dict = page.get_text("dict")
            page_width = page_dict.get("width", 612)  # Default US Letter width
            page_height = page_dict.get("height", 792)  # Default US Letter height
            
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            spans.append({
                                "text": text,
                                "font": span.get("font", "").lower(),
                                "size": round(span["size"], 2),
                                "bbox": [round(coord, 2) for coord in span["bbox"]],
                                "page": page_num,
                                "page_width": page_width,
                                "page_height": page_height
                            })
        return spans

    def detect_title(self, spans: List[Dict]) -> str:
        """Identify document title using multiple strategies"""
        if not spans:
            return ""
        
        # Strategy 1: Largest font on first page
        first_page = [s for s in spans if s.get("page", 1) == 1]
        if first_page:
            max_size = max(s["size"] for s in first_page)
            candidates = [
                s for s in first_page 
                if s["size"] >= self.title_font_threshold * max_size
                and len(s["text"].split()) >= self.min_heading_words
            ]
            
            # Prefer text near top of page
            candidates.sort(key=lambda x: x["bbox"][1])
            title = " ".join(s["text"].strip() for s in candidates[:3]).strip()
            if title:
                return title
        
        # Strategy 2: First non-empty span with sufficient length
        for span in spans:
            if len(span["text"].split()) >= self.min_heading_words:
                return span["text"].strip()
        
        return ""

    def is_potential_heading(self, text: str) -> bool:
        """Check if text has characteristics of a heading"""
        if not text.strip():
            return False
            
        # Common heading patterns
        patterns = [
            r"^(section|chapter|appendix)\s+\w+",  # "Chapter 1", "Appendix A"
            r"^\d+(\.\d+)*\s+\w+",  # "1.1 Introduction", "2.3.4 Results"
            r"^[A-Z][A-Z0-9\s]+$",  # ALL CAPS headings
            r"^[IVXLCDM]+\.?\s+\w+",  # Roman numerals
        ]
        
        return any(re.match(p, text.strip(), re.IGNORECASE) for p in patterns)

    def extract_outline(self, pdf_path: str) -> Dict:
        """Main extraction function with improved hierarchy handling"""
        try:
            doc = fitz.open(pdf_path)
            spans = self.extract_spans(doc)
            title = self.detect_title(spans)

            outline = []
            current_level = 1  # Tracks current hierarchy depth
            prev_heading = None
            
            for i, span in enumerate(spans):
                span_text = span["text"].strip()
                
                # Skip if matches title or too short
                if (normalize_text(span_text) == normalize_text(title) or 
                    len(span_text.split()) < self.min_heading_words):
                    continue
                
                # Extract features and predict
                feats = extract_features(span, spans[i - 1] if i > 0 else None)
                x = [[feats[k] for k in self.FEATURE_ORDER]]
                
                try:
                    label = self.le.inverse_transform(self.model.predict(x))[0]
                except:
                    continue

                if label in ("H1", "H2", "H3", "H4"):
                    level_num = int(label[1])
                    
                    # Maintain hierarchy consistency
                    if prev_heading:
                        max_allowed = prev_heading["level"] + 1
                        level_num = min(level_num, max_allowed)
                    
                    heading = {
                        "level": f"H{level_num}",
                        "text": span_text,
                        "page": span["page"],
                        "bbox": span["bbox"]
                    }
                    outline.append(heading)
                    prev_heading = {"level": level_num, "page": span["page"]}
                    current_level = level_num

            return {
                "title": title,
                "outline": outline,
                "source": os.path.basename(pdf_path)
            }
        
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return {
                "title": "",
                "outline": [],
                "source": os.path.basename(pdf_path),
                "error": str(e)
            }

    def process(self, input_dir="app/input", output_dir="app/output"):
        """Batch process all PDFs in directory"""
        os.makedirs(output_dir, exist_ok=True)
        stats = {"processed": 0, "errors": 0}
        
        for filename in tqdm(os.listdir(input_dir), desc="Processing PDFs"):
            if filename.lower().endswith(".pdf"):
                try:
                    result = self.extract_outline(os.path.join(input_dir, filename))
                    out_path = os.path.join(
                        output_dir, 
                        filename.replace(".pdf", ".json").replace(".PDF", ".json")
                    )
                    with open(out_path, "w") as jf:
                        json.dump(result, jf, indent=2)
                    stats["processed"] += 1
                except Exception as e:
                    print(f"Failed to process {filename}: {str(e)}")
                    stats["errors"] += 1
        
        print(f"\nProcessing complete. Success: {stats['processed']}, Errors: {stats['errors']}")

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    return re.sub(r"\s+", " ", text.strip().lower())

def extract_features(span: Dict, prev_span: Optional[Dict] = None) -> Dict:
    """Enhanced feature extraction with positional awareness"""
    text = span["text"]
    bbox = span["bbox"]
    page_height = span.get("page_height", 792)
    page_width = span.get("page_width", 612)
    
    # Calculate y-distance from previous span
    y_distance = 0
    if prev_span:
        prev_bbox = prev_span["bbox"]
        y_distance = abs(bbox[1] - prev_bbox[3])
        
        # Consider same line if y-difference is small
        if abs(bbox[1] - prev_bbox[1]) < 5:
            y_distance = 0
    
    # Calculate centering
    span_center = (bbox[0] + bbox[2]) / 2
    page_center = page_width / 2
    is_centered = int(abs(span_center - page_center) < 0.15 * page_width)
    
    # Numbered heading detection
    numbered = int(bool(re.match(r"^(\d+[\.\)]|[IVXLCDM]+\.?|•|\-)\s", text.strip())))
    
    features = {
        "font_size": span["size"],
        "is_bold": int("bold" in span.get("font", "")),
        "x0": bbox[0],
        "word_count": len(text.split()),
        "capital_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "ends_colon": int(text.strip().endswith(":")),
        "numbered": numbered,
        "y_distance": y_distance,
        "is_centered": is_centered,
        "starts_capital": int(text[0].isupper()) if text else 0,
        "all_caps": int(text.isupper()),
        "line_length": bbox[2] - bbox[0],
        "page_position": bbox[1] / page_height,
        "prev_font_size": prev_span["size"] if prev_span else 0,
        "prev_is_bold": int("bold" in prev_span.get("font", "")) if prev_span else 0
    }
    
    return features

if __name__ == "__main__":
    extractor = PDFOutlineExtractor()
    extractor.process()