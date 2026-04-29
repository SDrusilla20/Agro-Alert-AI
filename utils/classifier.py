"""
Alert severity classifier.

Tries to load a fine-tuned DistilBERT/RoBERTa sequence classification model
from `models/classifier_model/`. If unavailable, falls back to a rule-based
heuristic that scores severity from extracted weather entities + advisory text.

Labels:
    GREEN  - Safe / No action
    YELLOW - Moderate / Be aware
    ORANGE - High risk / Be prepared
    RED    - Extreme risk / Take action
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

CLF_MODEL_DIR = os.path.join("models", "classifier_model")

LEVELS = ["GREEN", "YELLOW", "ORANGE", "RED"]

LEVEL_META = {
    "GREEN": {
        "label": "Safe",
        "color": "#10b981",
        "bg": "rgba(16,185,129,0.15)",
        "description": "No significant weather threat. Continue routine farm operations.",
    },
    "YELLOW": {
        "label": "Moderate",
        "color": "#eab308",
        "bg": "rgba(234,179,8,0.15)",
        "description": "Be aware. Some weather parameters need monitoring.",
    },
    "ORANGE": {
        "label": "High Risk",
        "color": "#f97316",
        "bg": "rgba(249,115,22,0.15)",
        "description": "Be prepared. Conditions can damage crops without precaution.",
    },
    "RED": {
        "label": "Extreme Risk",
        "color": "#ef4444",
        "bg": "rgba(239,68,68,0.18)",
        "description": "Take action immediately. Severe agricultural impact expected.",
    },
}


@dataclass
class ClassificationResult:
    label: str
    probabilities: Dict[str, float]
    top_keywords: List[str]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "probabilities": self.probabilities,
            "top_keywords": self.top_keywords,
        }


# ---------------- Model loader ----------------


def _try_load_hf_classifier():
    if not os.path.isdir(CLF_MODEL_DIR):
        return None, None
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
        import torch  # noqa: F401

        tokenizer = AutoTokenizer.from_pretrained(CLF_MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(CLF_MODEL_DIR)
        model.eval()
        return tokenizer, model
    except Exception:
        return None, None


# ---------------- Heuristic fallback ----------------

_RISK_KEYWORDS = {
    "RED": [
        "extreme", "severe", "cyclone", "very heavy", "flood", "flash flood",
        "destructive", "catastrophic", "warning", "lashed",
    ],
    "ORANGE": [
        "heavy rainfall", "heavy rain", "thunderstorm", "lightning", "hailstorm",
        "strong wind", "gale", "lodging", "water logging", "waterlogging",
        "fungal", "pest outbreak",
    ],
    "YELLOW": [
        "moderate", "spraying", "advisory", "humidity", "monitor", "postpone",
        "precaution", "support",
    ],
    "GREEN": [
        "normal", "clear", "fair", "no rain", "stable", "favourable", "favorable",
    ],
}


def _keyword_score(text: str) -> Dict[str, float]:
    text_l = text.lower()
    scores = {lvl: 0.0 for lvl in LEVELS}
    for lvl, keywords in _RISK_KEYWORDS.items():
        for kw in keywords:
            if kw in text_l:
                scores[lvl] += 1.0
    return scores


def _entity_score(summary: dict) -> Dict[str, float]:
    scores = {lvl: 0.0 for lvl in LEVELS}

    def val(label):
        v = summary.get(label)
        return v["value"] if v and v.get("value") is not None else None

    tmax = val("TEMP_MAX")
    tmin = val("TEMP_MIN")
    rain = val("RAINFALL_LEVEL")
    hum = val("HUMIDITY")
    wind = val("WIND_SPEED")

    if tmax is not None:
        if tmax >= 42:
            scores["RED"] += 2.5
        elif tmax >= 38:
            scores["ORANGE"] += 2.0
        elif tmax >= 35:
            scores["YELLOW"] += 1.2
        else:
            scores["GREEN"] += 1.0

    if tmin is not None and tmin <= 5:
        scores["ORANGE"] += 1.5  # frost risk

    if rain is not None:
        if rain >= 115:
            scores["RED"] += 3.0
        elif rain >= 65:
            scores["ORANGE"] += 2.2
        elif rain >= 16:
            scores["YELLOW"] += 1.2
        else:
            scores["GREEN"] += 0.8

    if hum is not None:
        if hum >= 90:
            scores["ORANGE"] += 1.2
        elif hum >= 75:
            scores["YELLOW"] += 1.0
        else:
            scores["GREEN"] += 0.4

    if wind is not None:
        if wind >= 60:
            scores["RED"] += 2.0
        elif wind >= 40:
            scores["ORANGE"] += 1.5
        elif wind >= 25:
            scores["YELLOW"] += 0.8
        else:
            scores["GREEN"] += 0.4

    return scores


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    # Add a tiny prior so we never get all-zero
    base = {k: v + 0.05 for k, v in scores.items()}
    mx = max(base.values())
    exps = {k: math.exp(v - mx) for k, v in base.items()}
    s = sum(exps.values())
    return {k: round(v / s, 4) for k, v in exps.items()}


def _top_keywords(text: str, k: int = 6) -> List[str]:
    text_l = text.lower()
    found: List[str] = []
    for level in ["RED", "ORANGE", "YELLOW", "GREEN"]:
        for kw in _RISK_KEYWORDS[level]:
            if kw in text_l and kw not in found:
                found.append(kw)
            if len(found) >= k:
                return found
    return found


# ---------------- Public API ----------------


class SeverityClassifier:
    def __init__(self):
        self.tokenizer, self.model = _try_load_hf_classifier()
        self.using_model = self.model is not None

    def predict(self, text: str, entity_summary: Optional[dict] = None) -> ClassificationResult:
        if not text or not text.strip():
            return ClassificationResult(
                label="GREEN",
                probabilities={lvl: 0.25 for lvl in LEVELS},
                top_keywords=[],
            )

        # Try the fine-tuned HF model first
        if self.model is not None:
            try:
                import torch

                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                with torch.no_grad():
                    logits = self.model(**inputs).logits[0]
                probs = torch.softmax(logits, dim=-1).tolist()

                # Map to our 4-class scheme using id2label if available
                id2label = getattr(self.model.config, "id2label", {})
                mapped = {lvl: 0.0 for lvl in LEVELS}
                for idx, p in enumerate(probs):
                    raw = str(id2label.get(idx, idx)).upper()
                    for lvl in LEVELS:
                        if lvl in raw:
                            mapped[lvl] += float(p)
                            break
                # Re-normalize if mapping landed
                total = sum(mapped.values())
                if total > 0:
                    mapped = {k: round(v / total, 4) for k, v in mapped.items()}
                    label = max(mapped, key=mapped.get)
                    return ClassificationResult(
                        label=label,
                        probabilities=mapped,
                        top_keywords=_top_keywords(text),
                    )
            except Exception:
                pass

        # Heuristic fallback
        s_kw = _keyword_score(text)
        s_en = _entity_score(entity_summary or {})
        combined = {lvl: s_kw[lvl] + s_en[lvl] for lvl in LEVELS}
        probs = _softmax(combined)
        label = max(probs, key=probs.get)
        return ClassificationResult(
            label=label,
            probabilities=probs,
            top_keywords=_top_keywords(text),
        )
