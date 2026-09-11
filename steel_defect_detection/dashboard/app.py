"""
Jindal Stainless AI Surface Defect Inspection Platform
Team: GenCoders (Lipi Bagarti & Kartik Ranjan Singh)
Jindal Stainless Engineering Case Study Competition 2026

Features:
- Real-time multi-model ensemble detection (YOLO, Faster R-CNN, RT-DETR, EfficientNet-B0)
- Weighted Boxes Fusion (WBF) and Dual Confidence Fusion
- Decision Engine with False-Alarm Suppression and Severity Classification
- Live PLC / OPC-UA Telemetry Payload Inspector
- Database Logging and Defect Analytics
- 1-2 Slide Executive Summary Presentation
"""

import sys
import os
import site

# Ensure user site packages and project root are in sys.path
user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import time
import base64
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

from preprocessing.preprocess import preprocess_image, load_config
from backend.model_manager import ModelManager
from backend.inference_service import InferenceService
from database.database import init_db
from database.repository import get_statistics, get_inspections


st.set_page_config(
    page_title="Jindal Stainless AI Defect Detection | GenCoders",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern industrial styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        color: white;
    }
    .status-clean {
        background-color: #065F46;
        color: #34D399;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
    }
    .status-defect {
        background-color: #7F1D1D;
        color: #F87171;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
    }
    .executive-slide {
        background: #0B1120;
        border: 2px solid #2563EB;
        border-radius: 16px;
        padding: 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_service():
    config = load_config()
    db_path = config.get("database", {}).get("path", "database/inspections.db")
    init_db(str(ROOT / db_path))
    mm = ModelManager(config)
    mm.load_all_models()
    return InferenceService(config, mm)


def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/steel-i-beam.png", width=64)
        st.title("Jindal Stainless AI")
        st.caption("Surface Inspection & Quality Control System")
        st.divider()

        st.subheader("⚙️ Inspection Controls")
        conf_slider = st.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05)
        crop_padding = st.slider("Crop Padding Fraction", 0.0, 0.3, 0.1, 0.05)
        
        st.subheader("🤖 Model Weights (WBF)")
        w_yolo = st.slider("YOLO Weight", 0.1, 2.0, 1.0, 0.1)
        w_frcnn = st.slider("Faster R-CNN Weight", 0.1, 2.0, 1.0, 0.1)
        w_rtdetr = st.slider("RT-DETR Weight", 0.1, 2.0, 1.0, 0.1)
        
        st.divider()
        st.caption("Target Speed: >60 FPS | Industrial Strip Speed: 15–20 m/s")

    # Main Tabs
    tab1, tab2 = st.tabs([
        "🔍 Real-Time Inspection",
        "📊 Defect Analytics & DB",
    ])

    service = get_service()

    # ==========================================
    # TAB 1: REAL-TIME INSPECTION
    # ==========================================
    with tab1:
        st.header("🔩 Real-Time Steel Strip Defect Inspection")
        st.caption("Upload a steel strip surface image or select a sample benchmark image from NEU-DET.")

        col_up, col_sample = st.columns([2, 1])
        with col_up:
            uploaded_file = st.file_uploader(
                "Upload Strip Surface Image",
                type=["jpg", "jpeg", "png", "bmp"],
                help="Accepts high-resolution line-scan and area-scan camera images",
            )

        with col_sample:
            st.markdown("**Or Test with Sample Benchmark Defect:**")
            sample_defect = st.selectbox(
                "Select NEU-DET Sample",
                ["None", "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
            )

        image_bytes = None
        filename = "upload.jpg"

        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name
        elif sample_defect != "None":
            # Search for sample image in dataset
            sample_dir = ROOT / "data" / "yolo_format" / "images" / "val"
            sample_matches = list(sample_dir.glob(f"{sample_defect}*.jpg"))
            if sample_matches:
                sample_path = sample_matches[0]
                with open(sample_path, "rb") as f:
                    image_bytes = f.read()
                filename = sample_path.name
            else:
                st.warning(f"No sample found for {sample_defect} in val set.")

        if image_bytes:
            with st.spinner("Executing 4-Model Ensemble Pipeline + Decision Engine..."):
                # Apply dynamic slider thresholds
                service.ensemble.conf_threshold = conf_slider
                service.ensemble.crop_padding = crop_padding
                service.ensemble.model_weights = [w_yolo, w_frcnn, w_rtdetr]

                result = service.process_image(image_bytes, filename=filename, save_to_db=True)

            # Display Results Header
            st.divider()
            h_col1, h_col2, h_col3, h_col4 = st.columns(4)

            with h_col1:
                if result["final_decision"] == "CLEAN":
                    st.markdown('<div class="status-clean">🟢 STATUS: CLEAN</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-defect">🔴 STATUS: DEFECTIVE ({result["num_defects"]})</div>', unsafe_allow_html=True)

            with h_col2:
                action_color = "red" if result["overall_action"] in ["mark_reject", "line_stop"] else "orange" if result["overall_action"] == "operator_alert" else "green"
                st.metric("Action Triggered", result["overall_action"].upper())

            with h_col3:
                st.metric("Inference Latency", f"{result['latency']['inference_ms']} ms", f"{result['latency']['fps']} FPS")

            with h_col4:
                st.metric("DB Record ID", str(result["inspection_id"])[:8] if result["inspection_id"] else "Local")

            # Image Visualizer
            st.subheader("Inspection View")
            img_col1, img_col2 = st.columns(2)

            # Decode original and annotated
            orig_pil = Image.open(Path(filename) if Path(filename).exists() else uploaded_file if uploaded_file else sample_path)
            annotated_bytes = base64.b64decode(result["annotated_image_base64"])
            annotated_nparr = np.frombuffer(annotated_bytes, np.uint8)
            annotated_bgr = cv2.imdecode(annotated_nparr, cv2.IMREAD_COLOR)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            with img_col1:
                st.markdown("**Original Surface Image**")
                st.image(orig_pil, use_container_width=True)

            with img_col2:
                st.markdown("**AI Ensemble Annotations (WBF + EfficientNet)**")
                st.image(annotated_rgb, use_container_width=True)

            # Defect Details Table
            if result["detections"]:
                st.subheader("📋 Detected Defect Diagnostics")
                table_data = []
                for d in result["detections"]:
                    table_data.append({
                        "Defect Type": d.get("class_name", d.get("defect_type", "unknown")),
                        "Detector Conf (WBF)": f"{d.get('detector_confidence', 0.0):.1%}",
                        "Classifier Conf (EffNet)": f"{d.get('classifier_confidence', 0.0):.1%}",
                        "Fused Confidence": f"{d.get('final_confidence', 0.0):.1%}",
                        "Model Agreement": f"{d.get('model_agreement', 1)} / 3 Detectors",
                        "Severity": d.get("severity", "low").upper(),
                        "Action": d.get("action", "log").upper(),
                        "BBox [x1,y1,x2,y2]": [round(v, 1) for v in d.get("bbox", d.get("box", []))],
                    })
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)

                # PLC Telemetry Payload
                with st.expander("⚡ PLC / OPC-UA Output Payload (Industry 4.0 Standard)", expanded=False):
                    st.json(result["plc_payload"])
        else:
            st.info("👆 Please upload a steel surface image or select a benchmark sample above.")

    # ==========================================
    # TAB 2: DEFECT ANALYTICS & DATABASE
    # ==========================================
    with tab2:
        st.header("📊 Industrial Defect Quality Analytics")
        st.caption("Live aggregate metrics from the inspection database (SQLite/PostgreSQL schema)")

        try:
            stats = get_statistics()
            recent_records = get_inspections(limit=20)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Inspected Coils/Strips", stats.get("total_inspections", 0))
            c2.metric("Total Defects Flagged", stats.get("total_detections", 0))
            clean_count = stats.get("by_decision", {}).get("CLEAN", 0)
            defect_count = stats.get("by_decision", {}).get("DEFECTIVE", 0)
            yield_rate = (clean_count / (clean_count + defect_count) * 100) if (clean_count + defect_count) > 0 else 100.0
            c3.metric("First-Pass Yield (FPY)", f"{yield_rate:.1f}%")
            c4.metric("False Alarm Rate Target", "< 0.5%")

            st.divider()

            g1, g2 = st.columns(2)
            with g1:
                st.subheader("Defect Distribution by Type")
                by_type = stats.get("by_defect_type", {})
                if by_type:
                    fig_type = px.pie(
                        names=list(by_type.keys()),
                        values=list(by_type.values()),
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Bold,
                    )
                    st.plotly_chart(fig_type, use_container_width=True)
                else:
                    st.info("No defect data recorded yet.")

            with g2:
                st.subheader("Severity Breakdown")
                by_sev = stats.get("by_severity", {})
                if by_sev:
                    fig_sev = px.bar(
                        x=list(by_sev.keys()),
                        y=list(by_sev.values()),
                        labels={"x": "Severity Level", "y": "Defect Count"},
                        color=list(by_sev.keys()),
                        color_discrete_map={"low": "#10B981", "medium": "#F59E0B", "high": "#EF4444", "critical": "#7F1D1D"},
                    )
                    st.plotly_chart(fig_sev, use_container_width=True)
                else:
                    st.info("No severity data recorded yet.")

            st.subheader("Recent Inspection Log")
            if recent_records:
                st.dataframe(pd.DataFrame(recent_records), use_container_width=True)
            else:
                st.info("No inspection records logged yet.")

        except Exception as e:
            st.error(f"Error fetching analytics: {e}")


if __name__ == "__main__":
    main()

