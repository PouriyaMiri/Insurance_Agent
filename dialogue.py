import re
import difflib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from nlu import detect_intent, extract_entities
from premium import calculate_premium
from rag import DocChunk, RAGIndex


CURRENT_YEAR = datetime.now().year
CANCEL_WORDS = {"cancel", "stop", "nevermind", "never mind"}
EXIT_WORDS = {
    "hang up",
    "goodbye",
    "bye",
    "exit",
    "quit",
    "stop",
    "end",
    "terminate",
    "close",
    "disconnect",
}

_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "for": "4", 
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "ate": "8",  
    "nine": "9",
}
HUMAN_FALLBACK = (
    "I don't have enough information for more. "
    "You can say 'human agent' to be connected to our service center or check our website."
)

def _looks_like_insurance_request(t: str) -> bool:
    return any(
        p in t
        for p in [
            "need insurance",
            "want insurance",
            "looking for insurance",
            "get insurance",
            "buy insurance",
            "insurance for my car",
        ]
    )

def _spoken_digits_to_string(text: str) -> str:
    t = (text or "").lower()
    raw_digits = "".join(ch for ch in t if ch.isdigit())

    tokens = re.findall(r"[a-zA-Z]+|\d+", t)
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.isdigit():
            out.append(tok)
            i += 1
            continue

        if tok in {"double", "triple"} and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            d = _DIGIT_WORDS.get(nxt)
            if d:
                out.append(d * (2 if tok == "double" else 3))
                i += 2
                continue

        d = _DIGIT_WORDS.get(tok)
        if d:
            out.append(d)
        i += 1

    spoken = "".join(out)
    if len(spoken) >= len(raw_digits):
        return spoken
    return raw_digits


def _extract_policy_number(text: str) -> str:
    s = _spoken_digits_to_string(text)
    s = "".join(ch for ch in s if ch.isdigit())
    return s if len(s) >= 6 else ""


def _contains_any(text: str, phrases: set[str]) -> bool:
    t = (text or "").lower()
    return any(p in t for p in phrases)


def normalize_city(city: str) -> str:
    raw = (city or "").strip()
    c = raw.lower().strip()
    if not c:
        return raw

    lj_variants = {
        "ljublajan",
        "ljublijana",
        "ljubljana",
        "ljubljkana",
        "leobliana",
        "liubljana",
        "lubljana",
    }
    if c in lj_variants or "ljubl" in c:
        return "Ljubljana"

    if difflib.SequenceMatcher(None, c, "ljubljana").ratio() >= 0.75:
        return "Ljubljana"

    known = {
        "maribor": "Maribor",
        "celje": "Celje",
        "koper": "Koper",
    }
    if c in known:
        return known[c]
    
    return raw.strip()


def _first_sentence(text: str, max_len: int = 140) -> str:
    t = " ".join((text or "").strip().split())
    if not t:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", t)
    sent = parts[0] if parts else t
    if len(sent) > max_len:
        sent = sent[: max_len - 1].rstrip() + "…"
    return sent


def _is_bad_sentence(s: str) -> bool:
    s = (s or "").strip()
    return len(s) < 12 or s in [".", "…"] or bool(re.fullmatch(r"[^\w]+", s))


def _looks_like_pricing(t: str) -> bool:
    return any(k in t for k in ["pricing", "prcing", "price", "cost", "quote", "premium", "how much"])


def _looks_like_claim_info_only(t: str) -> bool:
    positives = ["claim", "claims", "claiming", "report", "reporting"]
    negatives = ["don't want to submit", "do not want to submit", "not submit", "not to submit", "just info", "information", "explain", "tell me"]
    return any(p in t for p in positives) and any(n in t for n in negatives)


COVERAGE_SUMMARY = {
    "basic": "Basic: third-party liability only (covers damage/injury you cause to others).",
    "standard": "Standard: liability + collision/own-damage (deductible applies).",
    "premium": "Premium: broadest cover (liability + collision + theft/fire/weather) and optional add-ons like roadside assistance.",
}


def _coverage_difference_answer() -> str:
    return (
        "Differences: "
        f"{COVERAGE_SUMMARY['basic']} "
        f"{COVERAGE_SUMMARY['standard']} "
        f"{COVERAGE_SUMMARY['premium']} "
        "Do you want the cheapest option or the most coverage?"
    )


def _is_dissatisfied(t: str) -> bool:
    bad = [
        "not precise",
        "irrelevant",
        "wrong",
        "useless",
        "stupid",
        "idiot",
        "you are not helping",
        "what are you talking about",
        "why you don't provide",
        "why dont you provide",
        "why don't answer",
        "why dont answer",
    ]
    return any(b in t for b in bad)


def _claim_missing_slots(slots: Dict) -> List[str]:
    needed = [
        "insurance_number",
        "injuries",
        "accident_city",
        "accident_date",
        "accident_description",
        "police_report",
        "vehicle_drivable",
        "third_party_involved",
    ]
    return [k for k in needed if k not in slots]


def _ask_claim_question(missing_key: str) -> str:
    if missing_key == "insurance_number":
        return "What is your insurance/policy number? (You can read it as it appears on your policy card.)"
    if missing_key == "injuries":
        return "Were there any injuries? Please say yes or no."
    if missing_key == "accident_city":
        return "Which city did the accident happen in?"
    if missing_key == "accident_date":
        return "What date did it happen? You can say 'today', 'yesterday', or a date like 2025-12-22."
    if missing_key == "accident_description":
        return "Briefly, what happened? One sentence is enough."
    if missing_key == "police_report":
        return "Was the police notified? Please say yes or no. If yes, do you have a report/reference number?"
    if missing_key == "vehicle_drivable":
        return "Is your car drivable right now? Please say yes or no."
    if missing_key == "third_party_involved":
        return "Were other vehicles involved? Please say yes or no."
    return "Can you tell me a bit more?"


def _parse_yes_no(text: str) -> Optional[bool]:
    t = (text or "").strip().lower()

    if t in ["np", "nop", "noo", "nahh"]:
        return False

    if re.search(r"\b(no|nope|nah|not)\b", t):
        return False
    if re.search(r"\b(yes|yeah|yep|y|sure|correct|ok)\b", t):
        return True

    return None


def _parse_date(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    today = datetime.now()

    if "today" in t:
        return today.strftime("%Y-%m-%d")
    if "yesterday" in t:
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")

    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def _extract_accident_location(text: str) -> Tuple[Optional[str], Optional[str]]:
    t = (text or "").strip()
    m = re.search(
        r"\bin\s+([A-Za-zÀ-ž\- ]+?)(?:,\s*([A-Za-zÀ-ž\- ]+))?(?:[.!?]|$)",
        t,
        flags=re.IGNORECASE,
    )
    if not m:
        return None, None

    part1 = m.group(1).strip()
    part2 = (m.group(2) or "").strip()

    if part2:
        return part1, normalize_city(part2)
    return None, normalize_city(part1)


def _claim_update_from_text(user_text: str, slots: Dict) -> None:
    tl = (user_text or "").lower().strip()

    area, city = _extract_accident_location(user_text)
    if city and "accident_city" not in slots:
        slots["accident_city"] = city
    if area and "accident_area" not in slots:
        slots["accident_area"] = area

    d = _parse_date(user_text)
    if d and "accident_date" not in slots:
        slots["accident_date"] = d

    yn = _parse_yes_no(user_text)
    if yn is None and len((user_text or "").strip()) > 8 and "accident_description" not in slots:
        if any(w in tl for w in ["accident", "crash", "collision", "rear-ended", "hit", "bumped"]):
            slots["accident_description"] = (user_text or "").strip()

    ref = re.search(
        r"\b(?:report|ref|reference|case)\s*#?\s*([A-Za-z0-9\-]+)\b",
        user_text or "",
        flags=re.IGNORECASE,
    )
    if ref:
        slots["police_report_ref"] = ref.group(1)


def _claim_apply_expected_answer(user_text: str, slots: Dict) -> None:
    expected = slots.get("claim_expected")
    if not expected:
        return

    yn = _parse_yes_no(user_text)
    d = _parse_date(user_text)

    if expected in ["injuries", "vehicle_drivable", "third_party_involved"]:
        if yn is not None:
            slots[expected] = yn
            slots.pop("claim_expected", None)
        return

    if expected == "police_report":
        if yn is not None:
            slots["police_report"] = yn
            if yn is True:
                ref = re.search(
                    r"\b(?:report|ref|reference|case)\s*#?\s*([A-Za-z0-9\-]+)\b",
                    user_text or "",
                    flags=re.IGNORECASE,
                )
                if ref:
                    slots["police_report_ref"] = ref.group(1)
            slots.pop("claim_expected", None)
        return

    if expected == "accident_date":
        if d:
            slots["accident_date"] = d
            slots.pop("claim_expected", None)
        return

    if expected == "accident_city":
        area, city = _extract_accident_location(user_text)
        if not city:
            candidate = re.split(r"[,.!?;]", (user_text or "").strip())[0].strip()
            if 2 <= len(candidate) <= 40 and not re.search(r"\d", candidate):
                city = normalize_city(candidate)
        if city:
            slots["accident_city"] = city
            if area:
                slots["accident_area"] = area
            slots.pop("claim_expected", None)
        return

    if expected in {"insurance_number", "policy_number"}:
        txt = (user_text or "").strip()

        pn = _extract_policy_number(txt)
        if pn:
            slots["insurance_number"] = pn
            slots.pop("claim_expected", None)
            return

        m = re.search(r"\b([A-Za-z0-9][A-Za-z0-9\-]{4,})\b", txt)
        if m:
            slots["insurance_number"] = m.group(1)
            slots.pop("claim_expected", None)
        return

    if expected == "accident_description":
        txt = (user_text or "").strip()
        if len(txt) > 5:
            slots["accident_description"] = txt
            slots.pop("claim_expected", None)
        return



def _generate_claim_number(state) -> str:
    now = datetime.now()
    yyyymm = now.strftime("%Y%m")

    counters = state.slots.get("claim_counters", {})
    if not isinstance(counters, dict):
        counters = {}

    n = int(counters.get(yyyymm, 0)) + 1
    counters[yyyymm] = n
    state.slots["claim_counters"] = counters

    return f"{yyyymm}{n:05d}"


def estimate_hp_from_engine_size(engine_size_l: float) -> Tuple[int, str]:
    if engine_size_l <= 1.0:
        return 95, "estimated from engine size"
    if engine_size_l <= 1.2:
        return 105, "estimated from engine size"
    if engine_size_l <= 1.4:
        return 120, "estimated from engine size"
    if engine_size_l <= 1.6:
        return 135, "estimated from engine size"
    if engine_size_l <= 1.8:
        return 155, "estimated from engine size"
    if engine_size_l <= 2.0:
        return 180, "estimated from engine size"
    return 200, "estimated from engine size"


class SessionState(BaseModel):
    slots: Dict[str, object] = Field(default_factory=dict)
    last_intent: Optional[str] = None
    turns: int = 0


class TurnResult(BaseModel):
    response_text: str
    end_call: bool = False


def _missing_premium_slots(slots: Dict) -> List[str]:
    needed = ["vehicle_age", "horsepower", "city", "coverage_level"]
    return [k for k in needed if k not in slots]


def _ask_one_missing(missing: List[str], slots: Dict) -> str:
    if not missing:
        return "What detail should we adjust?"
    k = missing[0]
    if k == "vehicle_age":
        return "What’s the vehicle year (e.g., 2010) or age in years?"
    if k == "horsepower":
        if "engine_size_l" in slots:
            return "I can estimate horsepower from engine size—do you want that, or do you know the exact HP?"
        return "About how many horsepower is the vehicle? If you don’t know, tell me engine size (e.g., 1.4 / 1.6)."
    if k == "city":
        return "Which city is the vehicle primarily used in?"
    if k == "coverage_level":
        return "Do you want basic, standard, or premium coverage?"
    return f"Could you tell me {k}?"


def _qa_answer_or_followup(user_text: str, docs: List[DocChunk], state: "SessionState") -> str:
    t = (user_text or "").lower().strip()
    state.slots["qa_turns"] = int(state.slots.get("qa_turns", 0)) + 1
    if _is_dissatisfied(t):
        state.slots["qa_frustration"] = int(state.slots.get("qa_frustration", 0)) + 1
    else:
        state.slots["qa_frustration"] = max(0, int(state.slots.get("qa_frustration", 0)) - 1)

    qa_turns = int(state.slots["qa_turns"])
    fr = int(state.slots.get("qa_frustration", 0))

    
    website = " You can also review details on our website for the full wording." if qa_turns == 1 else ""
    human = " If you’d rather speak to a person, type human." if (qa_turns >= 5 and fr >= 2) else ""

    if _looks_like_insurance_request(t):
        state.slots["force_intent"] = "premium_estimate"
        return "Sure — I can help you get car insurance. Let’s start with the vehicle year."


    if _looks_like_claim_info_only(t):
        docs = docs or []
        best = docs[0] if docs else None
        candidate = _first_sentence(best.text, max_len=140) if (best and best.text) else ""
        base = candidate if not _is_bad_sentence(candidate) else "Claims info: you can report online/phone, then provide incident details and evidence."
        return f"{base}{website} What part of claims do you want (steps, documents, timelines, or coverage)?{human}"

    if _looks_like_pricing(t):
        state.slots["force_intent"] = "premium_estimate"
        if "vehicle_age" not in state.slots and "vehicle_year" not in state.slots:
            return f"Sure — I can estimate pricing. What’s the vehicle year (e.g., 2010)?{human}"
        missing = _missing_premium_slots(state.slots)
        return f"Sure — I can estimate pricing. {_ask_one_missing(missing, state.slots)}{human}"

    if any(p in t for p in ["plans", "plan", "options", "offer", "coverage levels", "coverage level", "levels you offer"]):
        return f"We offer 3 coverage levels: Basic, Standard, and Premium.{website} Want the differences between them?{human}"

    if any(p in t for p in ["difference", "differences", "compare", "comparison", "what are the differences", "what is their differences"]):
        return _coverage_difference_answer() + human

    if any(p in t for p in ["how many", "number of", "how many models", "how many plans", "how many options"]):
        return f"We offer 3 options (Basic, Standard, Premium). Do you want the cheapest or the most coverage?{human}"

    if any(p in t for p in ["what are you talking about", "huh", "doesn't make sense", "irrelevant"]):
        return f"Got it — are you asking about coverage levels, pricing, or claim reporting?{human}"
    if not docs or not docs[0].text:
        return HUMAN_FALLBACK
    
    best = docs[0] if docs else None
    candidate = _first_sentence(best.text, max_len=140) if (best and best.text) else ""
    base = candidate if not _is_bad_sentence(candidate) else "I can answer that, but I need one detail first."

    if "deductible" in t:
        follow = "Which coverage level (basic, standard, premium)?"
    elif "exclusion" in t or "excluded" in t or "not covered" in t:
        follow = "Which situation (theft, drunk driving, intentional damage, etc.)?"
    elif "roadside" in t or "replacement" in t:
        follow = "Do you want roadside assistance and a replacement vehicle included?"
    else:
        follow = "What specific part do you want to know (coverage, deductible, exclusions, or claims)?"

    return f"{base}{website} {follow}{human}"


def dialogue_manager(user_text: str, state: SessionState, rag: RAGIndex) -> TurnResult:
    state.turns += 1
    t = (user_text or "").lower().strip()

    # Global exit: user can always end call
    if _contains_any(t, EXIT_WORDS):
        # also stop any active claim intake
        state.slots["in_claim_intake"] = False
        state.slots.pop("claim_expected", None)
        return TurnResult(response_text="Okay — ending the call. Goodbye!", end_call=True)

    ents = extract_entities(user_text, existing_slots=state.slots)
    state.slots.update(ents)

    if "city" in state.slots:
        state.slots["city"] = normalize_city(str(state.slots["city"]))

    if "vehicle_year" in state.slots:
        try:
            y = int(state.slots["vehicle_year"])
            if 1950 <= y <= CURRENT_YEAR:
                state.slots["vehicle_age"] = CURRENT_YEAR - y
        except Exception:
            pass

    if "vehicle_age" not in state.slots:
        m = re.search(r"\bage\s*(is)?\s*(\d{1,2})\b", t)
        if m:
            state.slots["vehicle_age"] = int(m.group(2))
        elif re.fullmatch(r"\d{1,2}", t):
            age = int(t)
            if 0 < age < 100 and state.last_intent == "premium_estimate":
                state.slots["vehicle_age"] = age


    intent_res = detect_intent(user_text, last_intent=state.last_intent)

    forced = state.slots.pop("force_intent", None)
    if forced:
        intent_res.intent = forced
        intent_res.confidence = 0.99

    if state.slots.get("in_claim_intake"):
        if _contains_any(t, CANCEL_WORDS):
            state.slots["in_claim_intake"] = False
            state.slots.pop("claim_expected", None)
            intent_res.intent = "doc_qa"
            intent_res.confidence = 0.6
        else:
            intent_res.intent = "report_claim"
            intent_res.confidence = 0.95

    shopping_signals = ["cheapest", "lowest", "option", "basic coverage", "liability", "minimum coverage"]
    if intent_res.intent == "doc_qa" and any(s in t for s in shopping_signals):
        intent_res.intent = "premium_estimate"
        intent_res.confidence = max(intent_res.confidence, 0.8)

    if intent_res.intent == "report_claim" and _looks_like_claim_info_only(t):
        intent_res.intent = "doc_qa"
        intent_res.confidence = 0.85

    if intent_res.intent == "doc_qa" and t in ["yes", "yes please", "ok", "okay", "sure", "da", "ya"]:
        if state.last_intent == "premium_estimate":
            if not _missing_premium_slots(state.slots):
                intent_res.intent = "compare_coverage"
                intent_res.confidence = 0.9
            else:

                intent_res.intent = "premium_estimate"
                intent_res.confidence = 0.9

    if state.last_intent == "premium_estimate" and intent_res.intent == "doc_qa":
        intent_res.intent = "premium_estimate"
        intent_res.confidence = max(intent_res.confidence, 0.75)


    if intent_res.intent != "doc_qa":
        state.slots["qa_turns"] = 0

    state.last_intent = intent_res.intent


    if intent_res.intent == "handoff_human" or _contains_any(t, {"human", "human agent", "call me back", "call back"}):
        return TurnResult(
            response_text="Okay — I’m transferring you to a human agent now. (Prototype: transfer simulated.)",
            end_call=True,
        )


    if intent_res.intent == "report_claim":
        state.slots["in_claim_intake"] = True

        _claim_update_from_text(user_text, state.slots)
        _claim_apply_expected_answer(user_text, state.slots)


        missing = _claim_missing_slots(state.slots)
        if missing:
            next_key = missing[0]
            state.slots["claim_expected"] = next_key
            return TurnResult(response_text=_ask_claim_question(next_key))

        claim_no = _generate_claim_number(state)
        state.slots["claim_number"] = claim_no
        state.slots["in_claim_intake"] = False
        state.slots.pop("claim_expected", None)

        area = state.slots.get("accident_area")
        city = state.slots.get("accident_city")
        loc = f"{area}, {city}" if area and city else (city or "the provided location")

        return TurnResult(
            response_text=(
                "Thanks — I’ve recorded your claim details.\n"
                "Next steps:\n"
                "1) If anyone is injured or there is danger, contact emergency services.\n"
                "2) Take photos of the scene, vehicle damage, plates, and road signs.\n"
                "3) Collect third-party and witness contacts if available.\n"
                "4) Keep receipts for towing/urgent costs.\n\n"
                f"Summary: Accident in {loc} on {state.slots.get('accident_date', 'unknown date')}. "
                f"Injuries: {'yes' if state.slots.get('injuries') else 'no'}.\n"
                f"Your claim number is {claim_no}."
            ),
            end_call=False,
        )

    if intent_res.intent == "premium_estimate":
        if "coverage_level" not in state.slots and "cheapest" in t:
            state.slots["coverage_level"] = "basic"



        hp_note = ""
        if "horsepower" not in state.slots and "engine_size_l" in state.slots:
            try:
                hp_est, note = estimate_hp_from_engine_size(float(state.slots["engine_size_l"]))
                state.slots["horsepower"] = hp_est
                hp_note = f"(Using ~{hp_est} HP {note}.) "
            except Exception:
                pass

        missing = _missing_premium_slots(state.slots)
        if missing:
            return TurnResult(response_text= hp_note + _ask_one_missing(missing, state.slots))

        if "engine_size_l" not in state.slots:
            m = re.search(r"\b(\d(?:\.\d)?)\s*(l|liter|litre)\b", t)
            if m:
                state.slots["engine_size_l"] = float(m.group(1))

        res = calculate_premium(
            vehicle_age=int(state.slots["vehicle_age"]),
            horsepower=int(state.slots["horsepower"]),
            city=str(state.slots["city"]),
            coverage_level=str(state.slots["coverage_level"]).lower(),
        )

        return TurnResult(
            response_text=(
                f"Based on a {state.slots['vehicle_age']}-year-old vehicle with {state.slots['horsepower']} HP in "
                f"{state.slots['city']} with {state.slots['coverage_level']} coverage, your estimate is about "
                f"€{res.monthly_eur} per month. Want to compare basic vs standard vs premium pricing?"
            ),
            end_call=False,
        )

    if state.last_intent == "premium_estimate" and "coverage_level" not in state.slots:
        if any(w in t for w in ["difference", "compare", "comparison", "what is the difference"]):
            return TurnResult(response_text=_coverage_difference_answer(), end_call=False)

    if state.last_intent in {"premium_estimate", "compare_coverage"} and intent_res.intent == "doc_qa":
        return TurnResult(response_text="Do you want to pick a coverage level (basic/standard/premium), or get a new quote with different details?", end_call=False,)
        

    if intent_res.intent == "compare_coverage":
        missing = _missing_premium_slots(state.slots)
        if missing:
            return TurnResult(response_text=_coverage_difference_answer())

        base_args = dict(
            vehicle_age=int(state.slots["vehicle_age"]),
            horsepower=int(state.slots["horsepower"]),
            city=str(state.slots["city"]),
        )
        results = {}
        for level in ["basic", "standard", "premium"]:
            res = calculate_premium(coverage_level=level, **base_args)
            results[level] = res.monthly_eur

        return TurnResult(
            response_text=(
                "Here’s a quick comparison:\n"
                f"- Basic (liability only): €{results['basic']} per month\n"
                f"- Standard: €{results['standard']} per month\n"
                f"- Premium (most coverage): €{results['premium']} per month\n\n"
                "Which one do you prefer?"
            ),
            end_call=False,
        )

    rag_query = user_text
    if any(w in t for w in ["plans", "options", "coverage", "levels"]):
        rag_query = "auto insurance coverage levels basic standard premium differences"
    elif "claim" in t:
        rag_query = "auto claim process steps required information evidence timeline"
    elif any(w in t for w in ["deductible", "exclusion", "excluded", "not covered"]):
        rag_query = user_text 

    docs = rag.retrieve(rag_query, top_k=3)
    return TurnResult(response_text=_qa_answer_or_followup(user_text, docs, state), end_call=False)
