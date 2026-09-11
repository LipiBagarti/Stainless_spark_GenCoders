# AI-Powered Surface Defect Detection — Steel Strip Inspection System
### Jindal Stainless Engineering Case Study Competition 2026

**Team Name:** GenCoders  
**Team Members:** Lipi Bagarti, Kartik Ranjan Singh  
**PoC Dataset:** NEU-DET Benchmark (1,800 images across 6 defect classes: *crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches*)

---

## 1. Project Overview & Deliverables

This repository provides an industrial-grade, end-to-end computer vision platform for real-time steel-strip surface defect detection, localization, classification, and severity triage.

### Key Components Implemented:
1. **Multi-Model Ensemble & WBF:**
   - **YOLOv8** (Fast single-stage detector)
   - **Faster R-CNN ResNet-50-FPN** (High-precision anchor detector)
   - **RT-DETR** (Transformer-based global context detector)
   - **Weighted Boxes Fusion (WBF):** Consolidates bounding boxes and scores multi-model agreement ($1, 2, \text{or } 3\text{ models}$).
   - **EfficientNet-B0:** Fine-grained texture verification classifier trained on 3,332 ground-truth defect crops.
   - **Dual Confidence Fusion:** $C_{final} = 0.60 \times C_{det} + 0.40 \times C_{cls}$.
2. **False-Alarm Suppression Decision Engine:**
   - 5-stage gating pipeline: Confidence thresholding $\to$ Minimum defect area filter $\to$ Strip ROI filter $\to$ Multi-frame temporal confirmation $\to$ Severity matrix.
3. **FastAPI High-Performance Backend:**
   - Preloaded singleton `ModelManager`, image upload/stream endpoints, analytics endpoints, and health monitoring.
4. **SQLite Quality Database Layer:**
   - `inspections` and `detections` tables with full coil traceability and aggregate defect analytics.
5. **PLC / SCADA Integration:**
   - Standardized OPC-UA/Industrial Ethernet JSON payloads ready for automated line markers and mill deceleration triggers.
6. **Streamlit Industrial HMI Dashboard:**
   - Live inspection demo, multi-model diagnostics, defect distribution charts, PLC telemetry viewer, and embedded **1-2 Slide Executive Summary**.
7. **Executive Summary Submission (1-2 Slides):**
   - [`docs/executive_summary_slides.md`](docs/executive_summary_slides.md)
   - [`docs/executive_summary_presentation.html`](docs/executive_summary_presentation.html) (Interactive 16:9 presentation deck ready for presentation/PDF export).

---

## 2. Repository Structure

```
steel_defect_detection/
├── backend/                     # FastAPI Backend Server & Routes
│   ├── main.py                  # API entry point & lifespan manager
│   ├── model_manager.py         # Singleton model cache (YOLO, Faster R-CNN, RT-DETR, Classifier)
│   ├── inference_service.py     # End-to-end inference & annotation coordinator
│   └── routes/                  # /inspect, /analytics, /models
├── configs/
│   ├── config.yaml              # Thresholds, WBF weights, fusion weights, database & PLC rules
│   └── data.yaml                # YOLO / RT-DETR dataset configuration
├── dashboard/
│   └── app.py                   # Streamlit HMI inspection & executive presentation dashboard
├── data/
│   ├── classifier_format/       # Ground-truth defect crops (3,332 train, 854 val)
│   └── yolo_format/             # YOLO format images & labels (train / val)
├── database/
│   ├── database.py              # SQLite connection & schema initialization
│   ├── models.py                # Schema definitions
│   └── repository.py            # CRUD operations & aggregate statistics
├── deployment/
│   └── export_onnx.py           # ONNX / TensorRT export scripts
├── docs/
│   ├── case_study_analysis.md   # Comprehensive 30-part engineering analysis
│   ├── executive_summary_slides.md # 1-2 slide executive pitch markdown
│   └── executive_summary_presentation.html # Standalone 16:9 presentation deck
├── ensemble/
│   ├── wbf.py                   # Weighted Boxes Fusion with fallback NMS
│   ├── confidence_fusion.py     # Dual confidence fusion
│   └── ensemble_detector.py     # 4-model inference pipeline orchestrator
├── integration/
│   └── plc_payload.py           # PLC / OPC-UA JSON payload generator
├── models/
│   ├── prediction_schema.py     # Standardized Prediction & FusedDetection dataclasses
│   ├── yolo_detector.py         # YOLO model wrapper
│   ├── faster_rcnn_detector.py  # Faster R-CNN wrapper
│   ├── rtdetr_detector.py       # RT-DETR wrapper
│   └── efficientnet_classifier.py # EfficientNet-B0 classifier wrapper
├── postprocessing/
│   └── decision_engine.py       # Staged false-alarm filter & severity classifier
├── preprocessing/
│   ├── convert_neudet_to_yolo.py# Dataset parser
│   ├── create_classifier_crops.py # Ground-truth crop extractor (Phase 2)
│   └── preprocess.py            # CLAHE, denoising, and normalization
├── training/
│   ├── train_yolo.py            # YOLOv8 fine-tuning script
│   ├── train_faster_rcnn.py     # PyTorch Faster R-CNN training script
│   ├── train_rtdetr.py          # Ultralytics RT-DETR training script
│   ├── train_classifier.py      # EfficientNet-B0 transfer learning script
│   └── evaluate.py              # Per-class evaluation and WBF parameter search
├── requirements.txt
└── README.md
```

---

## 3. Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Ground-Truth Classifier Crops
```bash
python preprocessing/create_classifier_crops.py
```

### 3. Launch the FastAPI Backend
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be available at: `http://localhost:8000/docs`

### 4. Launch the Streamlit HMI Dashboard
```bash
streamlit run dashboard/app.py
```
View the dashboard at: `http://localhost:8501`

---

## 4. Executive Summary for Judges (Team GenCoders)

| Pillar | Specification | GenCoders Engineering Decision |
| :--- | :--- | :--- |
| **Problem Formulation** | Real-time surface defect detection & triage | Object Detection + Bounding Box Localization + Fine-grained Classification |
| **Ensemble Strategy** | Multi-detector diversity | YOLOv8 (Speed) + Faster R-CNN (Anchors) + RT-DETR (Attention) + WBF |
| **False-Alarm Defense** | Prevent false line stops | EfficientNet crop verifier + Dual Confidence Fusion + 5-Stage Gate |
| **Throughput** | High-speed mill line (15–20 m/s) | 65+ FPS with ONNX Runtime / TensorRT acceleration |
| **Industrial Integration**| PLC / SCADA Readiness | Standardized OPC-UA JSON payloads + SQLite / PostgreSQL Traceability |
| **Competition Deck** | 1-2 Slide Executive Pitch | View in [`docs/executive_summary_presentation.html`](docs/executive_summary_presentation.html) or Tab 4 of the Streamlit App |
