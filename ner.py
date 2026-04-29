"""
Named Entity Recognition for IMD Agromet advisories.

Tries to load a fine-tuned HuggingFace token-classification model from
`models/ner_model/`. If that is unavailable (e.g. fresh deployment without
weights uploaded yet), it falls back to a robust regex-based extractor so the
app still runs end-to-end.

The fallback handles BOTH:
  * Quantitative IMD phrasing
        "maximum temperature is 36 to 38 degrees Celsius"
        "rainfall of 75 mm"
        "humidity of 82 percent"
        "wind speed of 35 to 45 km/hr"
  * Qualitative national bulletin phrasing
        "Heavy Rainfall very likely at isolated places"
        "Heat Wave conditions very likely"
        "Thunderstorm accompanied with gusty winds(40-50kmph)"
        "Hot and Humid weather very likely"
        "Warm Night conditions very likely"

Qualitative spans are converted to realistic numeric estimates so the
classifier, rules engine and dashboard cards all behave normally.

Entity labels:
    TEMP_MAX, TEMP_MIN, RAINFALL_LEVEL, HUMIDITY, WIND_SPEED
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

NER_MODEL_DIR = os.path.join("models", "ner_model")

ENTITY_LABELS = [
    "TEMP_MAX",
    "TEMP_MIN",
    "RAINFALL_LEVEL",
    "HUMIDITY",
    "WIND_SPEED",
]


@dataclass
class Entity:
    label: str
    text: str
    start: int
    end: int
    value: Optional[float] = None
    unit: Optional[str] = None
    confidence: float = 0.85

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "value": self.value,
            "unit": self.unit,
            "confidence": round(self.confidence, 3),
        }


def _try_load_hf_pipeline():
    """Try to load a fine-tuned HF token classification pipeline."""
    if not os.path.isdir(NER_MODEL_DIR):
        return None
    try:
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_DIR)
        model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_DIR)
        return pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )
    except Exception:
        return None


# ---------- Regex fallback ----------

_NUM = r"(\d+(?:\.\d+)?)"


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def _upper_of_range(value_str: str) -> Optional[float]:
    """Handle '36 to 38' / '40-50' style ranges; return the upper bound."""
    nums = re.findall(_NUM, value_str)
    if not nums:
        return None
    if len(nums) == 1:
        return _to_float(nums[0])
    return _to_float(nums[-1])


def _quantitative_entities(text: str) -> List[Entity]:
    """Numeric, IMD-style measurements with explicit units."""
    entities: List[Entity] = []

    patterns = [
        # Maximum temperature: "maximum temperature ... 36 to 38 °C"
        (
            "TEMP_MAX",
            re.compile(
                r"(maximum\s+temperature|max\.?\s*temp(?:erature)?|"
                r"day\s+temperature|mercury)"
                r"[^.\n]{0,80}?"
                r"(\d+(?:\.\d+)?(?:\s*(?:to|-|–)\s*\d+(?:\.\d+)?)?)"
                r"\s*(?:°|deg(?:rees)?)?\s*c(?:elsius)?\b",
                re.IGNORECASE,
            ),
            "°C",
        ),
        # Minimum temperature
        (
            "TEMP_MIN",
            re.compile(
                r"(minimum\s+temperature|min\.?\s*temp(?:erature)?|"
                r"night\s+temperature)"
                r"[^.\n]{0,80}?"
                r"(\d+(?:\.\d+)?(?:\s*(?:to|-|–)\s*\d+(?:\.\d+)?)?)"
                r"\s*(?:°|deg(?:rees)?)?\s*c(?:elsius)?\b",
                re.IGNORECASE,
            ),
            "°C",
        ),
        # Rainfall amount in mm
        (
            "RAINFALL_LEVEL",
            re.compile(
                r"(rainfall|rain|precipitation|shower)"
                r"[^.\n]{0,80}?"
                r"(\d+(?:\.\d+)?(?:\s*(?:to|-|–)\s*\d+(?:\.\d+)?)?)\s*mm",
                re.IGNORECASE,
            ),
            "mm",
        ),
        # Humidity percentage
        (
            "HUMIDITY",
            re.compile(
                r"(humidity|RH)"
                r"[^.\n]{0,80}?"
                r"(\d+(?:\.\d+)?(?:\s*(?:to|-|–)\s*\d+(?:\.\d+)?)?)"
                r"\s*(?:%|per\s*cent|percent)",
                re.IGNORECASE,
            ),
            "%",
        ),
        # Wind speed: "wind speed 35 to 45 km/hr" OR "gusty winds(40-50kmph)"
        (
            "WIND_SPEED",
            re.compile(
                r"(wind\s*speed|gusty\s*winds?|winds?|squall)"
                r"\s*[\(\:\-]?\s*"
                r"(\d+(?:\.\d+)?(?:\s*(?:to|-|–)\s*\d+(?:\.\d+)?)?)"
                r"\s*\)?\s*(?:km\s*p\s*h|km\s*/\s*hr|km\s*/\s*h|kmph|kph)\b",
                re.IGNORECASE,
            ),
            "km/hr",
        ),
    ]

    for label, pattern, unit in patterns:
        for m in pattern.finditer(text):
            value_str = m.group(2)
            num_start = m.start(2)
            num_end = m.end(2)

            # Extend to include the unit if present right after the number
            tail = text[num_end : num_end + 24]
            unit_match = re.match(
                r"\s*\)?\s*(?:°\s*c|deg(?:rees)?\s*c(?:elsius)?|mm|%|"
                r"per\s*cent|percent|km\s*p\s*h|km\s*/\s*hr|km\s*/\s*h|kmph|kph)",
                tail,
                re.IGNORECASE,
            )
            if unit_match:
                num_end += unit_match.end()

            value = _upper_of_range(value_str)
            entities.append(
                Entity(
                    label=label,
                    text=text[num_start:num_end].strip(),
                    start=num_start,
                    end=num_end,
                    value=value,
                    unit=unit,
                    confidence=0.85,
                )
            )

    return entities


# ---- Qualitative phrasing typical of national bulletins ----------------------
#
# Each entry: (label, regex, estimated numeric value, unit, base confidence)
# The estimated value lets the downstream classifier and rules engine
# behave the same as for explicit numeric mentions.

_QUALITATIVE_PATTERNS = [
    # Rainfall ----------------------------------------------------------------
    ("RAINFALL_LEVEL",
     re.compile(r"\bextremely\s+heavy\s+(?:rain(?:fall)?|shower)s?\b", re.IGNORECASE),
     200.0, "mm", 0.78),
    ("RAINFALL_LEVEL",
     re.compile(r"\bvery\s+heavy\s+(?:rain(?:fall)?|shower)s?\b", re.IGNORECASE),
     130.0, "mm", 0.78),
    ("RAINFALL_LEVEL",
     re.compile(r"\bheavy\s+(?:rain(?:fall)?|shower)s?\b", re.IGNORECASE),
     80.0, "mm", 0.78),
    ("RAINFALL_LEVEL",
     re.compile(r"\bmoderate\s+(?:rain(?:fall)?|shower)s?\b", re.IGNORECASE),
     25.0, "mm", 0.72),
    ("RAINFALL_LEVEL",
     re.compile(r"\b(?:light\s+(?:to\s+moderate\s+)?(?:rain(?:fall)?|shower)s?|"
                r"thunder\s*shower|isolated\s+shower)s?\b",
                re.IGNORECASE),
     10.0, "mm", 0.65),

    # Temperature -------------------------------------------------------------
    ("TEMP_MAX",
     re.compile(r"\bsevere\s+heat\s*wave\b", re.IGNORECASE),
     45.0, "°C", 0.78),
    ("TEMP_MAX",
     re.compile(r"\bheat\s*wave\s*conditions?\b|\bheat\s*wave\b", re.IGNORECASE),
     42.0, "°C", 0.78),
    ("TEMP_MAX",
     re.compile(r"\bhot\s+and\s+humid\s+weather\b|\bhot\s+weather\b",
                re.IGNORECASE),
     36.0, "°C", 0.65),
    ("TEMP_MIN",
     re.compile(r"\bwarm\s+night\s+conditions?\b|\bwarm\s+night\b",
                re.IGNORECASE),
     28.0, "°C", 0.70),
    ("TEMP_MIN",
     re.compile(r"\bcold\s*wave\b|\bsevere\s+cold\s*wave\b", re.IGNORECASE),
     4.0, "°C", 0.75),
    ("TEMP_MIN",
     re.compile(r"\bground\s+frost\b|\bfrost\s+conditions?\b", re.IGNORECASE),
     2.0, "°C", 0.78),

    # Humidity ----------------------------------------------------------------
    ("HUMIDITY",
     re.compile(r"\bhot\s+and\s+humid\b", re.IGNORECASE),
     82.0, "%", 0.65),
    ("HUMIDITY",
     re.compile(r"\bhigh\s+humidity\b|\bhumid\s+conditions?\b", re.IGNORECASE),
     80.0, "%", 0.62),

    # Wind --------------------------------------------------------------------
    # Squally / very strong winds without numbers
    ("WIND_SPEED",
     re.compile(r"\bsquall(?:y|s)?\b|\bvery\s+strong\s+winds?\b", re.IGNORECASE),
     65.0, "km/hr", 0.65),
    ("WIND_SPEED",
     re.compile(r"\bstrong\s+winds?\b", re.IGNORECASE),
     45.0, "km/hr", 0.60),
]


def _qualitative_entities(text: str) -> List[Entity]:
    entities: List[Entity] = []
    for label, pattern, est_value, unit, conf in _QUALITATIVE_PATTERNS:
        for m in pattern.finditer(text):
            entities.append(
                Entity(
                    label=label,
                    text=text[m.start():m.end()],
                    start=m.start(),
                    end=m.end(),
                    value=est_value,
                    unit=unit,
                    confidence=conf,
                )
            )
    return entities


def _dedupe_and_sort(entities: List[Entity]) -> List[Entity]:
    """Drop overlapping spans of the same label; keep highest confidence."""
    # Bucket by label first so different labels can coexist on overlapping text.
    by_label: dict = {}
    for e in entities:
        by_label.setdefault(e.label, []).append(e)

    kept: List[Entity] = []
    for label, ents in by_label.items():
        ents.sort(key=lambda e: (e.start, -e.confidence))
        last_end = -1
        for e in ents:
            if e.start >= last_end:
                kept.append(e)
                last_end = e.end
            # Else: overlap with a stronger earlier span — skip.

    kept.sort(key=lambda e: e.start)
    return kept


def _regex_entities(text: str) -> List[Entity]:
    return _dedupe_and_sort(
        _quantitative_entities(text) + _qualitative_entities(text)
    )


# ---------- Public API ----------


class NERExtractor:
    def __init__(self):
        self.pipeline = _try_load_hf_pipeline()
        self.using_model = self.pipeline is not None

    def extract(self, text: str) -> List[Entity]:
        """Return a list of Entity objects extracted from `text`."""
        if not text or not text.strip():
            return []

        if self.pipeline is not None:
            try:
                preds = self.pipeline(text)
                ents: List[Entity] = []
                for p in preds:
                    label = p.get("entity_group") or p.get("entity") or ""
                    label = label.replace("B-", "").replace("I-", "").upper()
                    if label not in ENTITY_LABELS:
                        continue
                    span = text[p["start"] : p["end"]]
                    nums = re.findall(_NUM, span)
                    value = _to_float(nums[-1]) if nums else None
                    ents.append(
                        Entity(
                            label=label,
                            text=span,
                            start=int(p["start"]),
                            end=int(p["end"]),
                            value=value,
                            confidence=float(p.get("score", 0.9)),
                        )
                    )
                if ents:
                    # Augment HF predictions with qualitative cues it might miss
                    return _dedupe_and_sort(ents + _qualitative_entities(text))
            except Exception:
                pass  # fall through to regex

        return _regex_entities(text)

    def to_summary(self, entities: List[Entity]) -> dict:
        """Reduce entities to the strongest single value per label."""
        summary: dict = {label: None for label in ENTITY_LABELS}

        # For each label keep the entity with the highest "severity" value
        # (i.e. the one that should drive recommendations) — fall back to
        # highest confidence if values tie or are missing.
        SEVERITY_PREFER_HIGH = {
            "TEMP_MAX": True,
            "TEMP_MIN": False,        # lower is more dangerous (frost)
            "RAINFALL_LEVEL": True,
            "HUMIDITY": True,
            "WIND_SPEED": True,
        }

        for label in ENTITY_LABELS:
            candidates = [e for e in entities if e.label == label]
            if not candidates:
                continue
            prefer_high = SEVERITY_PREFER_HIGH.get(label, True)

            def _key(e: Entity):
                v = e.value if e.value is not None else (
                    -1e9 if prefer_high else 1e9
                )
                return (v if prefer_high else -v, e.confidence)

            best = max(candidates, key=_key)
            summary[label] = {
                "value": best.value,
                "unit": best.unit,
                "text": best.text,
                "confidence": best.confidence,
            }
        return summary
