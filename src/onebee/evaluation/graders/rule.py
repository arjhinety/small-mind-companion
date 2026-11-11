from __future__ import annotations

import re
import string

_ABSTENTION_PHRASES: list[str] = [
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "i don't have that information",
    "i do not have that information",
    "you haven't told me",
    "you have not told me",
    "i don't recall",
    "i do not recall",
    "i cannot answer",
    "i can't answer",
    "i have no idea",
    "i'm not aware",
    "i am not aware",
    "i don't remember",
    "i do not remember",
    # Literal training-target phrases from generate_sft_data.py's abstention/
    # irrelevant_retrieval example generation -- without these, a model that learned to
    # reproduce these exact templates (the intended, correct behavior) was scored as NOT
    # abstaining, undercounting UAR. See docs/model_quirks.md's dedup-collapse entry and its
    # follow-up: this gap existed all along but was masked while abstention training examples
    # were themselves collapsed to ~1 by a separate dedup bug, so the model rarely reproduced
    # these exact strings; fixing the dedup bug increased verbatim-template output, which then
    # exposed this second, independent detector gap.
    "i don't think you've told me that",
    "i do not think you have told me that",
    "i don't have anything about that in what i remember",
    "i don't want to guess",
    "i do not want to guess",
    # Diversified paraphrases added when the fixed single-string templates were found to make
    # over-abstention too easy a shortcut to learn (docs/model_quirks.md #17,
    # docs/proper_scale_results.md's over-abstention tradeoff section).
    "doesn't ring a bell",
    "does not ring a bell",
    "not seeing anything about that",
    "don't think you've mentioned that",
    "do not think you have mentioned that",
    "rather not assume",
    "not something i have on record",
]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_text(prediction) == normalize_text(gold)


def contains_answer(response: str, gold: str) -> bool:
    """Whether the normalised gold answer occurs in the normalised response as a whole word or
    phrase. The rule behind ``pra_contain``, pre-registered in
    ``docs/preregistrations/study002-answer-containment.md``; change it only by amendment there."""
    needle = normalize_text(gold)
    if not needle:
        return False
    pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
    return re.search(pattern, normalize_text(response)) is not None


def fuzzy_match(prediction: str, gold: str, threshold: float = 0.8) -> bool:
    pred_tokens = set(normalize_text(prediction).split())
    gold_tokens = set(normalize_text(gold).split())
    if not pred_tokens and not gold_tokens:
        return True
    union = pred_tokens | gold_tokens
    intersection = pred_tokens & gold_tokens
    if not union:
        return False
    return len(intersection) / len(union) >= threshold


def entity_f1(prediction_entities: list[str], gold_entities: list[str]) -> dict[str, float]:
    pred_set = {e.lower().strip() for e in prediction_entities}
    gold_set = {e.lower().strip() for e in gold_entities}
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def detect_abstention(response: str) -> bool:
    lower = response.lower()
    return any(phrase in lower for phrase in _ABSTENTION_PHRASES)
