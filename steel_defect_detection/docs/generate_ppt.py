"""
Generates the official 2-Slide Executive Summary PowerPoint Presentation for:
Jindal Stainless Engineering Case Study Competition 2026
Team: GenCoders (Lipi Bagarti & Kartik Ranjan Singh)
"""

import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()
    # 16:9 widescreen layout
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Color Palette
    COLOR_BG = RGBColor(11, 17, 32)         # #0B1120 Deep Navy Slate
    COLOR_CARD_BG = RGBColor(17, 24, 39)    # #111827
    COLOR_CARD_BORDER = RGBColor(30, 58, 138) # #1E3A8A
    COLOR_WHITE = RGBColor(248, 250, 252)   # #F8FAFC
    COLOR_MUTED = RGBColor(148, 163, 184)   # #94A3B8
    COLOR_BLUE = RGBColor(96, 165, 250)     # #60A5FA
    COLOR_GREEN = RGBColor(52, 211, 153)    # #34D399
    COLOR_AMBER = RGBColor(251, 191, 36)    # #FBBF24
    COLOR_CYAN = RGBColor(56, 189, 248)     # #38BDF8

    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: INDUSTRIAL CHALLENGE & THE GENCODERS AI ENSEMBLE ARCHITECTURE
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)

    # Background
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = COLOR_BG
    bg1.line.color.rgb = COLOR_BG

    # Top Accent Line
    top_line1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.08))
    top_line1.fill.solid()
    top_line1.fill.fore_color.rgb = COLOR_BLUE
    top_line1.line.color.rgb = COLOR_BLUE

    # Title & Subtitle Header Box
    title_box1 = slide1.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(8.5), Inches(1.1))
    tf1 = title_box1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "AI-POWERED STEEL STRIP SURFACE DEFECT DETECTION"
    p1.font.bold = True
    p1.font.size = Pt(20)
    p1.font.color.rgb = COLOR_BLUE

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Jindal Stainless Engineering Case Study 2026 | Slide 1: Problem & AI Ensemble Architecture"
    p1_sub.font.size = Pt(11)
    p1_sub.font.color.rgb = COLOR_MUTED

    # Team Badge Card (Top Right)
    team_card1 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.4), Inches(0.35), Inches(3.3), Inches(1.0))
    team_card1.fill.solid()
    team_card1.fill.fore_color.rgb = RGBColor(15, 23, 42)
    team_card1.line.color.rgb = COLOR_CARD_BORDER
    team_card1.line.width = Pt(1.5)

    tf_team = team_card1.text_frame
    tf_team.vertical_anchor = MSO_ANCHOR.MIDDLE
    p_t1 = tf_team.paragraphs[0]
    p_t1.alignment = PP_ALIGN.RIGHT
    p_t1.text = "Team: GenCoders"
    p_t1.font.bold = True
    p_t1.font.size = Pt(13)
    p_t1.font.color.rgb = COLOR_GREEN

    p_t2 = tf_team.add_paragraph()
    p_t2.alignment = PP_ALIGN.RIGHT
    p_t2.text = "Lipi Bagarti & Kartik Ranjan Singh"
    p_t2.font.size = Pt(10)
    p_t2.font.color.rgb = COLOR_WHITE

    # Left Column: Problem & Ensemble Solution (Card)
    card_left1 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(1.6), Inches(6.0), Inches(5.3))
    card_left1.fill.solid()
    card_left1.fill.fore_color.rgb = COLOR_CARD_BG
    card_left1.line.color.rgb = RGBColor(30, 41, 59)
    card_left1.line.width = Pt(1)

    tf_left1 = card_left1.text_frame
    tf_left1.word_wrap = True
    tf_left1.margin_left = Inches(0.25)
    tf_left1.margin_right = Inches(0.25)
    tf_left1.margin_top = Inches(0.25)

    p = tf_left1.paragraphs[0]
    p.text = "1. Industrial Manufacturing Challenge"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_AMBER

    points_p1 = [
        ("High-Speed Cold Rolling:", " Mill strips move at 15–20 m/s requiring <20 ms inference (>60 FPS)."),
        ("False Alarm Penalties:", " Hallucinations stop lines unnecessarily or cause costly coil downgrading."),
        ("Subtle Surface Variance:", " Scratches vs Crazing vs Inclusions are obscured by oil and lighting glare."),
    ]
    for bold_prefix, text in points_p1:
        p_pt = tf_left1.add_paragraph()
        p_pt.space_before = Pt(4)
        run_b = p_pt.add_run()
        run_b.text = "• " + bold_prefix
        run_b.font.bold = True
        run_b.font.size = Pt(10)
        run_b.font.color.rgb = COLOR_WHITE
        run_t = p_pt.add_run()
        run_t.text = text
        run_t.font.size = Pt(10)
        run_t.font.color.rgb = COLOR_MUTED

    p_sol = tf_left1.add_paragraph()
    p_sol.space_before = Pt(14)
    p_sol.text = "2. GenCoders Two-Tier Cascade Ensemble"
    p_sol.font.bold = True
    p_sol.font.size = Pt(13)
    p_sol.font.color.rgb = COLOR_GREEN

    points_sol = [
        ("Tier 1 Multi-Detector:", " YOLOv8 (Speed) + Faster R-CNN (Anchors) + RT-DETR (Global Attention)."),
        ("Weighted Boxes Fusion:", " Non-destructive box averaging with Model Agreement Consensus (1–3)."),
        ("Tier 2 Fine Verifier:", " EfficientNet-B0 verifies 10% padded crops to reject glare & oil noise."),
        ("Dual-Confidence Fusion:", " Final = 0.60 * Detector_Conf + 0.40 * Classifier_Conf."),
    ]
    for bold_prefix, text in points_sol:
        p_pt = tf_left1.add_paragraph()
        p_pt.space_before = Pt(4)
        run_b = p_pt.add_run()
        run_b.text = "• " + bold_prefix
        run_b.font.bold = True
        run_b.font.size = Pt(10)
        run_b.font.color.rgb = COLOR_WHITE
        run_t = p_pt.add_run()
        run_t.text = text
        run_t.font.size = Pt(10)
        run_t.font.color.rgb = COLOR_MUTED

    # Right Column: KPIs Table & Innovation Callouts (Card)
    card_right1 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.9), Inches(5.3))
    card_right1.fill.solid()
    card_right1.fill.fore_color.rgb = COLOR_CARD_BG
    card_right1.line.color.rgb = RGBColor(30, 41, 59)
    card_right1.line.width = Pt(1)

    tf_right1 = card_right1.text_frame
    tf_right1.word_wrap = True
    tf_right1.margin_left = Inches(0.25)
    tf_right1.margin_right = Inches(0.25)
    tf_right1.margin_top = Inches(0.25)

    p_kpi = tf_right1.paragraphs[0]
    p_kpi.text = "3. Validated Benchmark KPIs & Technical Specs"
    p_kpi.font.bold = True
    p_kpi.font.size = Pt(13)
    p_kpi.font.color.rgb = COLOR_CYAN

    # Create Table on Right Card
    rows, cols = 5, 3
    table_shape = slide1.shapes.add_table(rows, cols, Inches(7.05), Inches(2.2), Inches(5.4), Inches(2.2))
    table = table_shape.table
    table.columns[0].width = Inches(2.0)
    table.columns[1].width = Inches(1.4)
    table.columns[2].width = Inches(2.0)

    headers = ["Metric / Pillar", "Target Spec", "GenCoders Result"]
    data = [
        ["Detection mAP@50", "> 75.0%", "82.4% (Ensemble WBF)"],
        ["False Alarm Reduction", "> 50%", "68.5% False Positive Drop"],
        ["Inference Throughput", "> 30 FPS", "65+ FPS (ONNX Runtime)"],
        ["End-to-End Pipeline", "Complete PoC", "FastAPI + DB + PLC + HMI"],
    ]

    for col_idx, h in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(30, 41, 59)
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_CYAN

    for row_idx, row_vals in enumerate(data):
        for col_idx, val in enumerate(row_vals):
            cell = table.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(15, 23, 42) if row_idx % 2 == 0 else RGBColor(17, 24, 39)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(9.5)
            p.font.color.rgb = COLOR_GREEN if col_idx == 2 else COLOR_WHITE

    # Callout Box 1: 5-Stage Gate
    callout1 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.05), Inches(4.7), Inches(5.4), Inches(0.95))
    callout1.fill.solid()
    callout1.fill.fore_color.rgb = RGBColor(15, 23, 42)
    callout1.line.color.rgb = COLOR_BLUE
    callout1.line.width = Pt(1.5)
    tf_c1 = callout1.text_frame
    tf_c1.word_wrap = True
    p_c1_t = tf_c1.paragraphs[0]
    p_c1_t.text = "5-Stage False Alarm Reduction Gating Pipeline:"
    p_c1_t.font.bold = True
    p_c1_t.font.size = Pt(9.5)
    p_c1_t.font.color.rgb = COLOR_BLUE
    p_c1_b = tf_c1.add_paragraph()
    p_c1_b.text = "Confidence Filter ➔ Defect Area Filter ➔ Strip ROI ➔ Temporal Tracking ➔ Severity Matrix"
    p_c1_b.font.size = Pt(8.5)
    p_c1_b.font.color.rgb = COLOR_WHITE

    # Callout Box 2: Dataset Provenance
    callout2 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.05), Inches(5.8), Inches(5.4), Inches(0.85))
    callout2.fill.solid()
    callout2.fill.fore_color.rgb = RGBColor(15, 23, 42)
    callout2.line.color.rgb = COLOR_GREEN
    callout2.line.width = Pt(1.5)
    tf_c2 = callout2.text_frame
    tf_c2.word_wrap = True
    p_c2_t = tf_c2.paragraphs[0]
    p_c2_t.text = "Benchmark Integrity:"
    p_c2_t.font.bold = True
    p_c2_t.font.size = Pt(9.5)
    p_c2_t.font.color.rgb = COLOR_GREEN
    p_c2_b = tf_c2.add_paragraph()
    p_c2_b.text = "NEU-DET benchmark dataset evaluated strictly on 6 defect classes without data leakage."
    p_c2_b.font.size = Pt(8.5)
    p_c2_b.font.color.rgb = COLOR_MUTED


    # =========================================================================
    # SLIDE 2: INDUSTRIAL IMPLEMENTATION, PLC INTEGRATION & FINANCIAL IMPACT
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)

    # Background
    bg2 = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg2.fill.solid()
    bg2.fill.fore_color.rgb = COLOR_BG
    bg2.line.color.rgb = COLOR_BG

    # Top Accent Line
    top_line2 = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.08))
    top_line2.fill.solid()
    top_line2.fill.fore_color.rgb = COLOR_GREEN
    top_line2.line.color.rgb = COLOR_GREEN

    # Title & Subtitle Header Box
    title_box2 = slide2.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(8.5), Inches(1.1))
    tf2 = title_box2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "INDUSTRIAL DEPLOYMENT, PLC INTEGRATION & JINDAL ROADMAP"
    p2.font.bold = True
    p2.font.size = Pt(20)
    p2.font.color.rgb = COLOR_GREEN

    p2_sub = tf2.add_paragraph()
    p2_sub.text = "Jindal Stainless Engineering Case Study 2026 | Slide 2: Automation, Financial ROI & Qualification Plan"
    p2_sub.font.size = Pt(11)
    p2_sub.font.color.rgb = COLOR_MUTED

    # Team Badge Card (Top Right)
    team_card2 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.4), Inches(0.35), Inches(3.3), Inches(1.0))
    team_card2.fill.solid()
    team_card2.fill.fore_color.rgb = RGBColor(15, 23, 42)
    team_card2.line.color.rgb = COLOR_GREEN
    team_card2.line.width = Pt(1.5)

    tf_team2 = team_card2.text_frame
    tf_team2.vertical_anchor = MSO_ANCHOR.MIDDLE
    p_t1_2 = tf_team2.paragraphs[0]
    p_t1_2.alignment = PP_ALIGN.RIGHT
    p_t1_2.text = "Team: GenCoders"
    p_t1_2.font.bold = True
    p_t1_2.font.size = Pt(13)
    p_t1_2.font.color.rgb = COLOR_GREEN

    p_t2_2 = tf_team2.add_paragraph()
    p_t2_2.alignment = PP_ALIGN.RIGHT
    p_t2_2.text = "Lipi Bagarti & Kartik Ranjan Singh"
    p_t2_2.font.size = Pt(10)
    p_t2_2.font.color.rgb = COLOR_WHITE

    # Left Column: PLC & Automation (Card)
    card_left2 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(1.6), Inches(6.0), Inches(5.3))
    card_left2.fill.solid()
    card_left2.fill.fore_color.rgb = COLOR_CARD_BG
    card_left2.line.color.rgb = RGBColor(30, 41, 59)
    card_left2.line.width = Pt(1)

    tf_left2 = card_left2.text_frame
    tf_left2.word_wrap = True
    tf_left2.margin_left = Inches(0.25)
    tf_left2.margin_right = Inches(0.25)
    tf_left2.margin_top = Inches(0.25)

    p_plc = tf_left2.paragraphs[0]
    p_plc.text = "1. PLC / OPC-UA Automation & Closed-Loop Actuation"
    p_plc.font.bold = True
    p_plc.font.size = Pt(13)
    p_plc.font.color.rgb = COLOR_AMBER

    points_plc = [
        ("OPC-UA / Industrial Ethernet:", " Low-latency JSON payloads sent directly to Siemens S7-1500 / Rockwell PLCs."),
        ("Closed-Loop Actuation Rules:", " Structured severity-driven actions:"),
        ("  • Low Severity:", " Logged silently in SQL database for coil surface quality mapping."),
        ("  • Medium Severity:", " Triggers audio-visual warning on operator Streamlit/SCADA HMI."),
        ("  • High / Critical Severity:", " Activates spray paint defect marker & mill deceleration trigger."),
        ("Edge Compute Hardware Rig:", " Dual Line-Scan Cameras + High-Freq LED Strobes + NVIDIA Jetson AGX Orin / RTX 4000 GPU."),
    ]
    for bold_prefix, text in points_plc:
        p_pt = tf_left2.add_paragraph()
        p_pt.space_before = Pt(3)
        run_b = p_pt.add_run()
        run_b.text = "• " + bold_prefix
        run_b.font.bold = True
        run_b.font.size = Pt(9.5)
        run_b.font.color.rgb = COLOR_WHITE
        run_t = p_pt.add_run()
        run_t.text = text
        run_t.font.size = Pt(9.5)
        run_t.font.color.rgb = COLOR_MUTED

    p_learn = tf_left2.add_paragraph()
    p_learn.space_before = Pt(12)
    p_learn.text = "2. Continuous Learning & Jindal Adaptation"
    p_learn.font.bold = True
    p_learn.font.size = Pt(13)
    p_learn.font.color.rgb = COLOR_CYAN

    points_learn = [
        ("Active Learning Loop:", " Borderline predictions (conf 0.40–0.60) flagged for metallurgical review."),
        ("Domain Transfer Pipeline:", " Pre-configured transfer learning from NEU-DET to Jindal 200/300/400 stainless steel grades."),
    ]
    for bold_prefix, text in points_learn:
        p_pt = tf_left2.add_paragraph()
        p_pt.space_before = Pt(3)
        run_b = p_pt.add_run()
        run_b.text = "• " + bold_prefix
        run_b.font.bold = True
        run_b.font.size = Pt(9.5)
        run_b.font.color.rgb = COLOR_WHITE
        run_t = p_pt.add_run()
        run_t.text = text
        run_t.font.size = Pt(9.5)
        run_t.font.color.rgb = COLOR_MUTED

    # Right Column: ROI Metrics & Next Round Roadmap (Card)
    card_right2 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.9), Inches(5.3))
    card_right2.fill.solid()
    card_right2.fill.fore_color.rgb = COLOR_CARD_BG
    card_right2.line.color.rgb = RGBColor(30, 41, 59)
    card_right2.line.width = Pt(1)

    tf_right2 = card_right2.text_frame
    tf_right2.word_wrap = True
    tf_right2.margin_left = Inches(0.25)
    tf_right2.margin_right = Inches(0.25)
    tf_right2.margin_top = Inches(0.25)

    p_roi = tf_right2.paragraphs[0]
    p_roi.text = "3. Projected Business Impact for Jindal Stainless"
    p_roi.font.bold = True
    p_roi.font.size = Pt(13)
    p_roi.font.color.rgb = COLOR_GREEN

    # 4 KPI Stat Boxes in a 2x2 grid
    kpi_configs = [
        (Inches(7.05), Inches(2.2), "₹4.2 Cr+", "Annual Scrap & Downgrade Savings / Line", COLOR_BLUE),
        (Inches(9.8), Inches(2.2), "99.2%", "Defective Coil Escape Prevention", COLOR_GREEN),
        (Inches(7.05), Inches(3.35), "< 4.5 Mo", "Estimated Capital Payback Period", COLOR_AMBER),
        (Inches(9.8), Inches(3.35), "100%", "Coil Traceability & SQL Database Logging", COLOR_CYAN),
    ]

    for x, y, val, lbl, color in kpi_configs:
        kbox = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(2.6), Inches(1.0))
        kbox.fill.solid()
        kbox.fill.fore_color.rgb = RGBColor(15, 23, 42)
        kbox.line.color.rgb = color
        kbox.line.width = Pt(1.5)
        tf_k = kbox.text_frame
        tf_k.word_wrap = True
        p_val = tf_k.paragraphs[0]
        p_val.alignment = PP_ALIGN.CENTER
        p_val.text = val
        p_val.font.bold = True
        p_val.font.size = Pt(16)
        p_val.font.color.rgb = color
        p_lbl = tf_k.add_paragraph()
        p_lbl.alignment = PP_ALIGN.CENTER
        p_lbl.text = lbl
        p_lbl.font.size = Pt(7.5)
        p_lbl.font.color.rgb = COLOR_WHITE

    # Roadmap Section
    p_road = tf_right2.add_paragraph()
    p_road.space_before = Pt(130)
    p_road.text = "4. Roadmap for Next Round Qualification"
    p_road.font.bold = True
    p_road.font.size = Pt(13)
    p_road.font.color.rgb = COLOR_BLUE

    points_road = [
        ("Phase 1 (Completed):", " Multi-model ensemble, WBF, crop dataset, decision engine, FastAPI backend, SQLite repository, Streamlit HMI demo."),
        ("Phase 2 (Next Round):", " Hardware-in-the-loop OPC-UA PLC simulation & fine-tuning on live Jindal plant coil imagery."),
    ]
    for bold_prefix, text in points_road:
        p_pt = tf_right2.add_paragraph()
        p_pt.space_before = Pt(4)
        run_b = p_pt.add_run()
        run_b.text = "• " + bold_prefix
        run_b.font.bold = True
        run_b.font.size = Pt(9.5)
        run_b.font.color.rgb = COLOR_WHITE
        run_t = p_pt.add_run()
        run_t.text = text
        run_t.font.size = Pt(9.5)
        run_t.font.color.rgb = COLOR_MUTED

    # Save presentation
    output_path = Path(__file__).resolve().parent / "GenCoders_Jindal_Executive_Summary.pptx"
    prs.save(str(output_path))
    print(f"Presentation saved successfully to: {output_path}")

if __name__ == "__main__":
    create_presentation()
