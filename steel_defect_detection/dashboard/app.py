"""
Jindal Stainless AI Surface Defect Inspection Platform
Real-Time Steel Strip Quality Assurance & Telemetry System
"""

import sys
import os
import site
from pathlib import Path

# Ensure user site packages and project root are in sys.path
user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)

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
    page_title="Jindal Stainless AI Defect Detection",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Industrial CSS Styling
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 4px;
        color: #60A5FA;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Status Badges */
    .status-badge-clean {
        background: linear-gradient(135deg, #065F46, #047857);
        border: 1px solid #10B981;
        color: #ECFDF5;
        padding: 10px 22px;
        border-radius: 10px;
        font-weight: 800;
        font-size: 1.25rem;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    .status-badge-defect {
        background: linear-gradient(135deg, #7F1D1D, #991B1B);
        border: 1px solid #EF4444;
        color: #FEF2F2;
        padding: 10px 22px;
        border-radius: 10px;
        font-weight: 800;
        font-size: 1.25rem;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }

    /* Action Badges */
    .action-badge {
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        text-transform: uppercase;
    }
    .action-pass { background-color: #064E3B; color: #34D399; border: 1px solid #059669; }
    .action-alert { background-color: #78350F; color: #FBBF24; border: 1px solid #D97706; }
    .action-reject { background-color: #7F1D1D; color: #F87171; border: 1px solid #DC2626; }
    .action-stop { background-color: #4C0519; color: #FDA4AF; border: 1px solid #E11D48; }

    /* Custom View Containers */
    .view-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
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
        st.markdown("### 🔩 Jindal Stainless AI")
        st.caption("Surface Inspection & Quality Control Platform")
        st.divider()

        st.subheader("🤖 AI Model Architecture")
        model_mode_option = st.selectbox(
            "Select Inspection Mode",
            [
                "4-Model Cascade Ensemble (WBF + EfficientNet) [Best Accuracy]",
                "RT-DETR Vision Transformer (Global Context)",
                "Faster R-CNN ResNet-50 (Anchor Precision)",
                "YOLOv8 Single-Stage (Ultra Fast)",
            ],
            index=0,
        )
        
        mode_key_map = {
            "4-Model Cascade Ensemble (WBF + EfficientNet) [Best Accuracy]": "ensemble",
            "RT-DETR Vision Transformer (Global Context)": "rtdetr",
            "Faster R-CNN ResNet-50 (Anchor Precision)": "faster_rcnn",
            "YOLOv8 Single-Stage (Ultra Fast)": "yolo_only",
        }
        selected_mode = mode_key_map[model_mode_option]

        st.subheader("⚙️ Inspection Thresholds")
        conf_slider = st.slider("Confidence Gate", 0.10, 0.90, 0.35, 0.05)
        crop_padding = st.slider("Crop Context Padding", 0.0, 0.30, 0.10, 0.05)
        
        if selected_mode == "ensemble":
            st.subheader("⚖️ WBF Fusion Weights")
            w_yolo = st.slider("YOLOv8 Weight", 0.1, 2.0, 1.0, 0.1)
            w_frcnn = st.slider("Faster R-CNN Weight", 0.1, 2.0, 1.0, 0.1)
            w_rtdetr = st.slider("RT-DETR Weight", 0.1, 2.0, 1.0, 0.1)
        else:
            w_yolo, w_frcnn, w_rtdetr = 1.0, 1.0, 1.0
        
        st.divider()
        st.caption("⚡ Target Speed: >60 FPS | Mill Line Speed: 15–20 m/s")

    # Main Header & Tabs
    tab1, tab2 = st.tabs([
        "🔍 Real-Time Surface Inspection",
        "📊 Defect Analytics & Quality DB",
    ])

    service = get_service()

    # ==========================================
    # TAB 1: REAL-TIME INSPECTION
    # ==========================================
    with tab1:
        st.markdown("## 🔍 Steel Strip Surface Inspection")
        st.caption("Real-time defect detection, classification, severity scoring, and PLC telemetry generation.")

        col_up, col_sample = st.columns([2, 1])
        with col_up:
            uploaded_file = st.file_uploader(
                "Upload Steel Strip Image (Area-Scan / Line-Scan)",
                type=["jpg", "jpeg", "png", "bmp"],
                help="Accepts standard high-resolution steel strip images",
            )

        with col_sample:
            st.markdown("**Or Test with Sample Benchmark Defect:**")
            sample_defect = st.selectbox(
                "Select Defect Benchmark Image",
                ["None", "scratches", "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale"],
            )

        image_bytes = None
        filename = "upload.jpg"

        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name
        elif sample_defect != "None":
            sample_dir = ROOT / "data" / "yolo_format" / "images" / "val"
            sample_matches = list(sample_dir.glob(f"{sample_defect}*.jpg"))
            if sample_matches:
                sample_path = sample_matches[0]
                with open(sample_path, "rb") as f:
                    image_bytes = f.read()
                filename = sample_path.name
            else:
                st.warning(f"No sample found for {sample_defect} in validation set.")

        if image_bytes:
            with st.spinner(f"Executing {model_mode_option.split('[')[0].strip()} + Decision Engine..."):
                # Configure service parameters
                service.ensemble.conf_threshold = conf_slider
                service.ensemble.crop_padding = crop_padding
                service.ensemble.model_weights = [w_yolo, w_frcnn, w_rtdetr]

                try:
                    result = service.process_image(
                        image_bytes=image_bytes,
                        filename=filename,
                        save_to_db=True,
                        model_mode=selected_mode,
                    )
                except TypeError:
                    result = service.process_image(
                        image_bytes=image_bytes,
                        filename=filename,
                        save_to_db=True,
                    )

            st.divider()

            # KPI Summary Header
            k1, k2, k3, k4 = st.columns(4)

            with k1:
                if result["final_decision"] == "CLEAN":
                    st.markdown('<div class="status-badge-clean">🟢 CLEAN / PRIME</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-badge-defect">🔴 DEFECTIVE ({result["num_defects"]})</div>', unsafe_allow_html=True)

            with k2:
                action = result.get("overall_action", "log").upper()
                action_class = "action-stop" if action == "LINE_STOP" else "action-reject" if action == "MARK_REJECT" else "action-alert" if action == "OPERATOR_ALERT" else "action-pass"
                st.markdown(f'<div class="action-badge {action_class}">Action: {action}</div>', unsafe_allow_html=True)
                st.caption("Closed-loop PLC actuation rule")

            with k3:
                st.metric("Latency & Speed", f"{result['latency']['inference_ms']} ms", f"{result['latency']['fps']} FPS")

            with k4:
                st.metric("Inspection Record ID", str(result.get("inspection_id", "local"))[:8] if result.get("inspection_id") else "Local Log")

            st.write("")

            # Visual Comparison
            img_col1, img_col2 = st.columns(2)

            # Decode original and annotated
            orig_pil = Image.open(Path(filename) if Path(filename).exists() else uploaded_file if uploaded_file else sample_path)
            annotated_bytes = base64.b64decode(result["annotated_image_base64"])
            annotated_nparr = np.frombuffer(annotated_bytes, np.uint8)
            annotated_bgr = cv2.imdecode(annotated_nparr, cv2.IMREAD_COLOR)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            with img_col1:
                st.markdown("#### 📷 Original Surface Image")
                st.image(orig_pil, use_container_width=True)

            with img_col2:
                st.markdown("#### 🎯 AI Defect Localization & Classification")
                st.image(annotated_rgb, use_container_width=True)

            # Defect Diagnostics Table
            if result["detections"]:
                st.markdown("### 📋 Defect Diagnostics & Telemetry Breakdown")
                table_data = []
                for d in result["detections"]:
                    table_data.append({
                        "Defect Type": d.get("class_name", d.get("defect_type", "unknown")).upper(),
                        "Detector Conf (WBF)": f"{d.get('detector_confidence', 0.0):.1%}",
                        "Classifier Conf (EffNet)": f"{d.get('classifier_confidence', 0.0):.1%}",
                        "Fused Final Score": f"{d.get('final_confidence', 0.0):.1%}",
                        "Model Agreement": f"{d.get('model_agreement', 1)} / 3 Models",
                        "Severity Level": d.get("severity", "low").upper(),
                        "PLC Action": d.get("action", "log").upper(),
                        "Bounding Box [x1, y1, x2, y2]": [round(v, 1) for v in d.get("bbox", d.get("box", []))],
                    })
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)

                # PLC / OPC-UA Telemetry Expander
                with st.expander("⚡ Industrial PLC / OPC-UA JSON Telemetry Payload", expanded=False):
                    st.json(result["plc_payload"])
            else:
                st.success("✨ No surface defects detected. The stainless steel strip meets prime surface quality standards.")
        else:
            st.info("👆 Please upload a steel surface image or choose a benchmark defect from the dropdown above.")

    # ==========================================
    # TAB 2: DEFECT ANALYTICS & DATABASE
    # ==========================================
    with tab2:
        st.markdown("## 📊 Defect Quality Analytics & Historical Database")
        st.caption("Live aggregate metrics and digital traceability from the SQL database.")

        try:
            stats = get_statistics()
            recent_records = get_inspections(limit=25)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Inspected Strips", stats.get("total_inspections", 0))
            c2.metric("Total Defect Detections", stats.get("total_detections", 0))
            clean_count = stats.get("by_decision", {}).get("CLEAN", 0)
            defect_count = stats.get("by_decision", {}).get("DEFECTIVE", 0)
            total = clean_count + defect_count
            yield_rate = (clean_count / total * 100) if total > 0 else 100.0
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
                        hole=0.45,
                        color_discrete_sequence=px.colors.qualitative.Bold,
                    )
                    fig_type.update_layout(margin=dict(t=20, b=20, l=20, r=20))
                    st.plotly_chart(fig_type, use_container_width=True)
                else:
                    st.info("No defect data logged yet.")

            with g2:
                st.subheader("Severity Breakdown")
                by_sev = stats.get("by_severity", {})
                if by_sev:
                    fig_sev = px.bar(
                        x=list(by_sev.keys()),
                        y=list(by_sev.values()),
                        labels={"x": "Severity Level", "y": "Count"},
                        color=list(by_sev.keys()),
                        color_discrete_map={"low": "#10B981", "medium": "#F59E0B", "high": "#EF4444", "critical": "#7F1D1D"},
                    )
                    fig_sev.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
                    st.plotly_chart(fig_sev, use_container_width=True)
                else:
                    st.info("No severity data logged yet.")

            st.markdown("### 📋 Recent Inspection Log")
            if recent_records:
                st.dataframe(pd.DataFrame(recent_records), use_container_width=True)
            else:
                st.info("No inspection records logged yet.")

        except Exception as e:
            st.error(f"Error fetching analytics: {e}")


if __name__ == "__main__":
    main()
