import re
from typing import Dict, Optional

def normalize_text(text: str) -> str:
    """Normalize text by removing extra spaces and converting to lowercase."""
    return re.sub(r"\s+", " ", text.strip().lower())

def extract_features(span: Dict, prev_span: Optional[Dict] = None) -> Dict:
    """
    Extract features from text span for classification.
    
    Args:
        span: Current text span with metadata
        prev_span: Previous span for contextual features
        
    Returns:
        Dictionary of features for ML model
    """
    text = span["text"]
    bbox = span["bbox"]
    
    # Calculate y-distance from previous span if available
    y_distance = 0
    if prev_span:
        prev_bbox = prev_span["bbox"]
        y_distance = abs(bbox[1] - prev_bbox[3])
        
        # Handle same line detection
        if abs(bbox[1] - prev_bbox[1]) < 5:  # Considered same line
            y_distance = 0
    
    # Calculate centering (assuming US Letter page width of 612)
    page_center = 612 / 2
    span_center = (bbox[0] + bbox[2]) / 2
    is_centered = int(abs(span_center - page_center) < 20)
    
    features = {
        # Formatting features
        "font_size": span["size"],
        "is_bold": int("bold" in span.get("font", "")),
        # Text features
        "word_count": len(text.split()),
        "capital_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "ends_colon": int(text.strip().endswith(":")),
        "numbered": int(bool(re.match(r"^(\d+[\.\)]|•|\-)\s", text.strip()))),
        "starts_capital": int(text[0].isupper()) if text else 0,
        "all_caps": int(text.isupper()),
        # Positional features
        "x0": bbox[0],
        "y_distance": y_distance,
        "is_centered": is_centered,
        "line_length": bbox[2] - bbox[0],
        "page_position": bbox[1] / 792,  # Normalized y-position
        # Contextual features
        "prev_font_size": prev_span["size"] if prev_span else 0,
        "prev_is_bold": int("bold" in prev_span.get("font", "")) if prev_span else 0
    }
    
    return features