"""
Generate a PDF report containing the input advisory, extracted entities,
alert level, English recommendations, and translated recommendations.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


_LEVEL_COLORS = {
    "GREEN": colors.HexColor("#10b981"),
    "YELLOW": colors.HexColor("#eab308"),
    "ORANGE": colors.HexColor("#f97316"),
    "RED": colors.HexColor("#ef4444"),
}


def build_report_pdf(
    advisory_text: str,
    entity_summary: Dict,
    alert_level: str,
    alert_probabilities: Dict[str, float],
    recommendations: List[Dict],
    translated_lines: Optional[List[str]] = None,
    target_language: Optional[str] = None,
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="AgroAlert AI Report",
    )

    styles = getSampleStyleSheet()
    h_title = ParagraphStyle(
        "h_title",
        parent=styles["Title"],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    h_sub = ParagraphStyle(
        "h_sub",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14,
    )
    h_section = ParagraphStyle(
        "h_section",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "body",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#1e293b"),
    )
    badge_style = ParagraphStyle(
        "badge",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        textColor=colors.white,
        alignment=TA_LEFT,
    )

    story = []

    # ---- Header ----
    story.append(Paragraph("AgroAlert AI — Advisory Report", h_title))
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}", h_sub
        )
    )

    # ---- Alert badge ----
    color = _LEVEL_COLORS.get(alert_level, colors.HexColor("#475569"))
    badge_tbl = Table(
        [[Paragraph(f"<b>Alert Level:&nbsp;&nbsp;{alert_level}</b>", badge_style)]],
        colWidths=[60 * mm],
    )
    badge_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), color),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("ROUNDEDCORNERS", [4, 4, 4, 4]),
            ]
        )
    )
    story.append(badge_tbl)
    story.append(Spacer(1, 6))

    # Probabilities
    if alert_probabilities:
        prob_rows = [["Severity", "Probability"]]
        for k in ["GREEN", "YELLOW", "ORANGE", "RED"]:
            if k in alert_probabilities:
                prob_rows.append([k, f"{alert_probabilities[k] * 100:.1f}%"])
        prob_tbl = Table(prob_rows, colWidths=[60 * mm, 40 * mm])
        prob_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.white, colors.HexColor("#f8fafc")]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(prob_tbl)

    # ---- Advisory text ----
    story.append(Paragraph("Original Advisory", h_section))
    safe_text = (advisory_text or "(empty)").replace("\n", "<br/>")
    story.append(Paragraph(safe_text, body))

    # ---- Extracted entities ----
    story.append(Paragraph("Extracted Weather Entities", h_section))
    rows = [["Parameter", "Value", "Unit", "Confidence"]]
    pretty = {
        "TEMP_MAX": "Max Temperature",
        "TEMP_MIN": "Min Temperature",
        "RAINFALL_LEVEL": "Rainfall",
        "HUMIDITY": "Humidity",
        "WIND_SPEED": "Wind Speed",
    }
    for key, label in pretty.items():
        info = entity_summary.get(key)
        if info:
            rows.append([
                label,
                str(info.get("value") if info.get("value") is not None else info.get("text", "—")),
                info.get("unit") or "—",
                f"{info.get('confidence', 0) * 100:.0f}%",
            ])
        else:
            rows.append([label, "—", "—", "—"])

    ent_tbl = Table(rows, colWidths=[55 * mm, 40 * mm, 30 * mm, 35 * mm])
    ent_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#f8fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(ent_tbl)

    # ---- Recommendations ----
    story.append(Paragraph("Recommended Actions", h_section))
    if not recommendations:
        story.append(Paragraph("No specific actions required.", body))
    else:
        for i, r in enumerate(recommendations, 1):
            story.append(
                Paragraph(
                    f"<b>{i}. {r.get('title', '')}</b> "
                    f"<font color='#64748b'>[{r.get('priority', '')} • "
                    f"{r.get('category', '')}]</font>",
                    body,
                )
            )
            story.append(Paragraph(r.get("detail", ""), body))
            story.append(Spacer(1, 4))

    # ---- Translation ----
    if translated_lines and target_language:
        story.append(Paragraph(f"Recommendations — {target_language}", h_section))
        for line in translated_lines:
            if line.strip():
                story.append(Paragraph(line, body))
                story.append(Spacer(1, 2))

    story.append(Spacer(1, 14))
    story.append(
        Paragraph(
            "<font size=8 color='#94a3b8'>Generated by AgroAlert AI · "
            "Always cross-check with the official IMD Agromet bulletin "
            "before taking field action.</font>",
            body,
        )
    )

    doc.build(story)
    return buf.getvalue()
