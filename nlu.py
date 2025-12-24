import re
from pydantic import BaseModel
from typing import Dict, Literal, Optional

Intent = Literal["doc_qa", "report_claim", "premium_estimate", "clarification", "handoff_human", "compare_coverage"]

class IntentResult(BaseModel):
    intent: Intent
    confidence: float

COVERAGE_ALIASES = {
    "basic": ["basic", "minimum", "liability", "low", "budget", "economy", "starter", "cheapest"],
    "standard": ["standard", "normal", "regular", "medium", "typical", "mid-tier", "moderate"],
    "premium": ["premium", "full", "comprehensive", "high", "best", "top-tier", "ultimate"],
}

keywords = {
    "doc_qa": ["what is", "tell me about", "information on", "details about", "explain", "how to"],
    "report_claim": ["report a claim", "file a claim", "accident", "crash", "stolen", "theft", "claim"],
    "premium_estimate": [
        "premium", "quote", "cost", "price", "how much", "estimate", " need insurance", "want insurance", "need a quote", "get a quote",
        "cheapest insurance", "insurance cost", "insurance premium", "insurance quote", "insurance price",
        "option", "options", "cheap option", "cheap options", "cheapest option", "lowest price", "need a quote", "get a quote",
    ],
    "handoff_human": ["agent", "representative", "call me back", "human", "talk to someone", "speak to someone", "customer service", "real person", "live person", "operator"],
    "compare_coverage": ["compare", "comparison", "compare them","show options", "show differences", "yes compare", "compare plans", "options"],

}

def detect_intent(user_text: str, last_intent: Optional[str] = None) -> IntentResult:
    t = user_text.lower().strip()

    if last_intent == "premium_estimate" and any(k in t for k in keywords["premium_estimate"]):
        return IntentResult(intent="premium_estimate", confidence=0.85)

    if any(k in t for k in keywords["compare_coverage"]):
        return IntentResult(intent="compare_coverage", confidence=0.9)


    if any(k in t for k in keywords["premium_estimate"]):
        return IntentResult(intent="premium_estimate", confidence=0.8)

    if any(k in t for k in keywords["report_claim"]):
        return IntentResult(intent="report_claim", confidence=0.8)

    if any(k in t for k in keywords["handoff_human"]):
        return IntentResult(intent="handoff_human", confidence=0.7)

    return IntentResult(intent="doc_qa", confidence=0.6)


def _extract_int(pattern: str, text: str) -> Optional[int]:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def extract_entities(user_text: str, existing_slots: Optional[Dict] = None) -> Dict:
    existing_slots = existing_slots or {}
    t = user_text.strip()
    tl = t.lower()

    out: Dict = {}

    for level, words in COVERAGE_ALIASES.items():
        if any(w in tl for w in words):
            out["coverage_level"] = level
            break


    hp = _extract_int(r"(\d+)\s*(?:hp|horsepower)", t)
    if hp is not None:
        out["horsepower"] = hp


    m = re.search(r"\b(?:engine\s*size\s*(?:is|=)\s*)?(\d\.\d)\s*(?:l|litre|liter)?\b", tl)
    if m:
        try:
            es = float(m.group(1))
            if 0.8 <= es <= 8.0:
                out["engine_size_l"] = es
        except Exception:
            pass


    m = re.search(r"\b(?:in|city)\s+([A-Za-zÀ-ž\- ]{2,})", t)
    if m:
        city = re.split(r"[,.!?;]", m.group(1).strip())[0].strip()
        if city.lower() in ["ljublajan", "ljublijana", "ljubljana"]:
            city = "Ljubljana"
        out["city"] = city

    m = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", tl)
    if m:
        out["vehicle_year"] = int(m.group(1))



    bare_number = _extract_int(r"^\s*(\d{1,4})\s*$", t)
    if bare_number is not None:
        if 1950 <= bare_number <= 2025:
            out["vehicle_year"] = bare_number
        elif 0 < bare_number <= 60:
            out["vehicle_age"] = bare_number

    age = _extract_int(r"(\d+)\s*(?:year|yr)[s]?\s*(?:old)?", t)
    if age is not None:
        out["vehicle_age"] = age

    return out
