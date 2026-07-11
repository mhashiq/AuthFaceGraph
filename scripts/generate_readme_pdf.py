from __future__ import annotations

import re
from datetime import datetime
from html import escape
from pathlib import Path
from typing import List, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"
OUTPUT_PATH = ROOT / "AUTHFACEGRAPH_AI_Technical_Report.pdf"


def build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=34,
            textColor=colors.HexColor("#0f172a"),
            alignment=TA_CENTER,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#334155"),
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverMeta",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#475569"),
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=12,
            spaceAfter=8,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading3",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            textColor=colors.HexColor("#334155"),
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBullet",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            leftIndent=12,
            firstLineIndent=0,
            bulletIndent=0,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportCode",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=8.5,
            leading=11,
            backColor=colors.HexColor("#f8fafc"),
            borderPadding=6,
            borderWidth=0.6,
            borderColor=colors.HexColor("#cbd5e1"),
            textColor=colors.HexColor("#0f172a"),
            leftIndent=0,
            rightIndent=0,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportTOC",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TOCEntry1",
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            leftIndent=10,
            firstLineIndent=-10,
            spaceBefore=2,
            textColor=colors.HexColor("#0f172a"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="TOCEntry2",
            fontName="Helvetica",
            fontSize=9.2,
            leading=12,
            leftIndent=18,
            firstLineIndent=-8,
            textColor=colors.HexColor("#475569"),
        )
    )
    return styles


def draw_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(colors.HexColor("#0f172a"))
    canvas.rect(0, height - 12 * mm, width, 12 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(16 * mm, height - 8 * mm, "AUTHFACEGRAPH AI Technical Report")
    canvas.setFont("Helvetica", 8.5)
    canvas.drawRightString(width - 16 * mm, height - 8 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.setLineWidth(0.4)
    canvas.line(16 * mm, 16 * mm, width - 16 * mm, 16 * mm)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(16 * mm, 10.5 * mm, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    canvas.restoreState()


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(16 * mm, 20 * mm, A4[0] - 32 * mm, A4[1] - 34 * mm, id="normal")
        self.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=draw_page)])

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and hasattr(flowable, "_heading_level"):
            level = getattr(flowable, "_heading_level")
            text = flowable.getPlainText()
            key = getattr(flowable, "_bookmark", None)
            if key:
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, level=level - 1, closed=False)
            self.notify("TOCEntry", (level, text, self.page))


def normalize_text(lines: List[str]) -> str:
    text = " ".join(part.strip() for part in lines if part.strip())
    return escape(text)


def is_table_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", line.strip()))


def parse_table(lines: List[str]) -> List[List[str]]:
    rows: List[List[str]] = []
    for raw in lines:
        cells = [escape(cell.strip()) for cell in raw.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def clean_code_block(text: str) -> str:
    return text.rstrip("\n")


def render_markdown_blocks(markdown_text: str, styles: dict[str, ParagraphStyle]):
    lines = markdown_text.splitlines()
    blocks = []
    i = 0
    skip_toc = False

    while i < len(lines):
        line = lines[i]

        if line.startswith("## Table of Contents"):
            skip_toc = True
            i += 1
            while i < len(lines) and not lines[i].startswith("---"):
                i += 1
            continue

        if skip_toc and line.startswith("---"):
            skip_toc = False
            i += 1
            continue

        if skip_toc:
            i += 1
            continue

        if not line.strip():
            i += 1
            continue

        heading_match = re.match(r"^(#{1,3})\s+(.*)$", line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            style_name = f"SectionHeading{level}"
            para = Paragraph(escape(title), styles[style_name])
            para._heading_level = level  # type: ignore[attr-defined]
            para._bookmark = f"heading_{len(blocks)}"  # type: ignore[attr-defined]
            blocks.append(para)
            blocks.append(Spacer(1, 2))
            i += 1
            continue

        if line.startswith("```"):
            fence = line.strip()[3:].strip()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            code_text = clean_code_block("\n".join(code_lines))
            blocks.append(Preformatted(code_text, styles["ReportCode"], dedent=0))
            blocks.append(Spacer(1, 2))
            continue

        if line.strip() == "$$":
            equation_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                equation_lines.append(lines[i])
                i += 1
            i += 1
            equation_text = clean_code_block("\n".join(equation_lines))
            blocks.append(Paragraph("<font face='Courier'>" + escape(equation_text).replace("\n", "<br/>") + "</font>", styles["ReportBody"]))
            blocks.append(Spacer(1, 2))
            continue

        if line.startswith("|"):
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            if len(table_lines) >= 2 and is_table_separator(table_lines[1]):
                table_rows = parse_table([table_lines[0]] + table_lines[2:])
            else:
                table_rows = parse_table(table_lines)
            if table_rows:
                tbl = Table(table_rows, repeatRows=1, hAlign="LEFT")
                tbl.setStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.8),
                        ("LEADING", (0, 0), (-1, -1), 11),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
                blocks.append(tbl)
                blocks.append(Spacer(1, 6))
            continue

        if re.match(r"^\s*([-*]|\d+\.)\s+", line):
            items = []
            ordered = bool(re.match(r"^\s*\d+\.\s+", line))
            while i < len(lines) and re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
                bullet_line = re.sub(r"^\s*([-*]|\d+\.)\s+", "", lines[i]).strip()
                items.append(Paragraph(escape(bullet_line), styles["ReportBullet"]))
                i += 1
            blocks.append(
                ListFlowable(
                    [ListItem(item) for item in items],
                    bulletType="1" if ordered else "bullet",
                    start="1",
                    leftPadding=12,
                )
            )
            blocks.append(Spacer(1, 2))
            continue

        paragraph_lines = [line]
        i += 1
        while i < len(lines):
            next_line = lines[i]
            if not next_line.strip():
                break
            if next_line.startswith(("#", "```", "|")) or re.match(r"^\s*([-*]|\d+\.)\s+", next_line) or next_line.strip() == "$$":
                break
            paragraph_lines.append(next_line)
            i += 1
        paragraph_text = normalize_text(paragraph_lines)
        if paragraph_text:
            blocks.append(Paragraph(paragraph_text, styles["ReportBody"]))
            blocks.append(Spacer(1, 1))

    return blocks


def build_story(styles: dict[str, ParagraphStyle]):
    readme_text = README_PATH.read_text(encoding="utf-8")
    story = []

    story.append(Spacer(1, 20 * mm))
    story.append(Paragraph("AUTHFACEGRAPH AI", styles["ReportTitle"]))
    story.append(Paragraph("AuthBrain AI Face Analysis Engine", styles["ReportSubtitle"]))
    story.append(
        Paragraph(
            "Professional technical report and implementation reference for the real-time facial behavior analysis platform.",
            styles["CoverMeta"],
        )
    )
    story.append(Spacer(1, 8 * mm))
    story.append(HRFlowable(width="68%", thickness=1.1, color=colors.HexColor("#0f172a"), spaceBefore=2, spaceAfter=8, hAlign="CENTER"))

    overview_box = Table(
        [
            [Paragraph("<b>Focus</b><br/>Real-time webcam facial analysis, emotion recognition, graph reasoning, and explainability.", styles["ReportBody"]), Paragraph("<b>Stack</b><br/>React, FastAPI, MediaPipe, OpenCV, PyTorch, PyTorch Geometric, SQLAlchemy.", styles["ReportBody"])],
            [Paragraph("<b>Scope</b><br/>Architecture, pipeline, models, rules, security, testing, and deployment.", styles["ReportBody"]), Paragraph("<b>Output</b><br/>A production-ready PDF suitable for handover, documentation, and research presentation.", styles["ReportBody"])],
        ],
        colWidths=[80 * mm, 80 * mm],
        hAlign="CENTER",
    )
    overview_box.setStyle(
        [
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
    )
    story.append(overview_box)
    story.append(PageBreak())

    toc_title = Paragraph("Table of Contents", styles["ReportTOC"])
    toc = TableOfContents()
    toc.levelStyles = [styles["TOCEntry1"], styles["TOCEntry2"], styles["TOCEntry2"]]
    story.append(toc_title)
    story.append(Spacer(1, 4))
    story.append(toc)
    story.append(PageBreak())

    story.extend(render_markdown_blocks(readme_text, styles))
    return story


def main() -> int:
    if not README_PATH.exists():
        raise FileNotFoundError(f"Missing README at {README_PATH}")

    styles = build_styles()
    story = build_story(styles)

    doc = ReportDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="AUTHFACEGRAPH AI Technical Report",
        author="GitHub Copilot",
        subject="Professional technical report generated from README",
        keywords="AUTHFACEGRAPH AI, face analysis, FastAPI, React, GNN, computer vision",
    )
    doc.multiBuild(story)
    print(f"Created {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())