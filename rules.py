"""
Rule-based crop recommendation engine.

Reads the entity summary produced by NERExtractor.to_summary() plus the
predicted alert level, then returns a structured list of recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Recommendation:
    title: str
    detail: str
    category: str  # IRRIGATION, PEST, DRAINAGE, MECHANICAL, GENERAL
    priority: str  # LOW, MEDIUM, HIGH, CRITICAL
    triggered_by: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "detail": self.detail,
            "category": self.category,
            "priority": self.priority,
            "triggered_by": self.triggered_by,
        }


def _val(summary: dict, key: str) -> Optional[float]:
    v = summary.get(key)
    if v and v.get("value") is not None:
        return float(v["value"])
    return None


def generate_recommendations(
    entity_summary: dict,
    alert_level: str = "GREEN",
) -> List[Recommendation]:
    recs: List[Recommendation] = []

    tmax = _val(entity_summary, "TEMP_MAX")
    tmin = _val(entity_summary, "TEMP_MIN")
    rain = _val(entity_summary, "RAINFALL_LEVEL")
    hum = _val(entity_summary, "HUMIDITY")
    wind = _val(entity_summary, "WIND_SPEED")

    # ---- Rainfall ----
    if rain is not None:
        if rain >= 65:
            recs.append(Recommendation(
                title="Open field drainage channels",
                detail=(
                    "Heavy rainfall expected. Clear field bunds and drainage "
                    "channels to prevent water logging in low-lying paddy and "
                    "vegetable plots."
                ),
                category="DRAINAGE",
                priority="HIGH" if rain < 115 else "CRITICAL",
                triggered_by=[f"Rainfall {rain:g} mm"],
            ))
            recs.append(Recommendation(
                title="Postpone fertilizer & pesticide spraying",
                detail=(
                    "Avoid all spraying operations until at least 24 hours "
                    "after rainfall subsides to prevent washoff and reduced "
                    "efficacy."
                ),
                category="GENERAL",
                priority="HIGH",
                triggered_by=[f"Rainfall {rain:g} mm"],
            ))
            recs.append(Recommendation(
                title="Monitor for fungal infections",
                detail=(
                    "Wet foliage from heavy rain combined with humidity can "
                    "trigger blight and mildew in tomato, chili, and grapes. "
                    "Inspect daily and pre-stage approved fungicide."
                ),
                category="PEST",
                priority="MEDIUM",
                triggered_by=[f"Rainfall {rain:g} mm"],
            ))
        elif rain >= 16:
            recs.append(Recommendation(
                title="Light field drainage check",
                detail=(
                    "Moderate rainfall expected. Walk fields after first "
                    "shower and clear any standing water spots."
                ),
                category="DRAINAGE",
                priority="MEDIUM",
                triggered_by=[f"Rainfall {rain:g} mm"],
            ))

    # ---- Humidity ----
    if hum is not None and hum > 75:
        recs.append(Recommendation(
            title="Pest & fungal watch (high humidity)",
            detail=(
                "Humidity above 75% favors aphids, whiteflies, leaf curl "
                "virus and powdery mildew. Increase scouting frequency and "
                "consider neem-based prophylactic spray during dry windows."
            ),
            category="PEST",
            priority="HIGH" if hum >= 85 else "MEDIUM",
            triggered_by=[f"Humidity {hum:g}%"],
        ))

    # ---- Heat ----
    if tmax is not None:
        if tmax >= 38:
            recs.append(Recommendation(
                title="Heat stress prevention & life-saving irrigation",
                detail=(
                    "Schedule short-duration irrigation in the early morning "
                    "or late evening. Apply mulch around horticultural crops "
                    "to retain soil moisture and reduce surface temperature."
                ),
                category="IRRIGATION",
                priority="HIGH" if tmax < 42 else "CRITICAL",
                triggered_by=[f"Max temp {tmax:g}°C"],
            ))
        elif tmax >= 35:
            recs.append(Recommendation(
                title="Schedule supplemental irrigation",
                detail=(
                    "Provide a light irrigation cycle to standing crops to "
                    "buffer against rising afternoon temperatures."
                ),
                category="IRRIGATION",
                priority="MEDIUM",
                triggered_by=[f"Max temp {tmax:g}°C"],
            ))

    # ---- Cold / frost ----
    if tmin is not None and tmin <= 5:
        recs.append(Recommendation(
            title="Frost protection measures",
            detail=(
                "Light irrigation in the late evening, smoke generation in "
                "orchards, and row covers on nursery beds can limit frost "
                "damage to tender crops."
            ),
            category="GENERAL",
            priority="HIGH",
            triggered_by=[f"Min temp {tmin:g}°C"],
        ))

    # ---- Wind ----
    if wind is not None:
        if wind >= 40:
            recs.append(Recommendation(
                title="Provide mechanical support to tall crops",
                detail=(
                    "Stake or earth-up banana, sugarcane, papaya and tomato "
                    "to prevent lodging from strong winds. Harvest mature "
                    "fruits ahead of the wind window where feasible."
                ),
                category="MECHANICAL",
                priority="HIGH" if wind < 60 else "CRITICAL",
                triggered_by=[f"Wind {wind:g} km/hr"],
            ))
        elif wind >= 25:
            recs.append(Recommendation(
                title="Inspect crop staking & trellises",
                detail=(
                    "Walk the field and reinforce loose stakes, trellises "
                    "and shade nets before wind speeds peak."
                ),
                category="MECHANICAL",
                priority="MEDIUM",
                triggered_by=[f"Wind {wind:g} km/hr"],
            ))

    # ---- Severity escalation ----
    if alert_level == "RED" and not any(r.priority == "CRITICAL" for r in recs):
        recs.append(Recommendation(
            title="Coordinate with local agriculture office",
            detail=(
                "Severe weather alert is in effect. Contact the nearest "
                "Krishi Vigyan Kendra (KVK) for district-specific contingency "
                "and crop insurance guidance."
            ),
            category="GENERAL",
            priority="CRITICAL",
            triggered_by=["Severity: RED"],
        ))
    elif alert_level == "GREEN" and not recs:
        recs.append(Recommendation(
            title="Continue routine farm operations",
            detail=(
                "Weather parameters look favourable. Proceed with planned "
                "intercultural operations and monitor next 24-hour bulletin."
            ),
            category="GENERAL",
            priority="LOW",
            triggered_by=["Severity: GREEN"],
        ))

    # Sort by priority
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    recs.sort(key=lambda r: order.get(r.priority, 4))
    return recs
