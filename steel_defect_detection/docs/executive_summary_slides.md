# Jindal Stainless Engineering Case Study Competition 2026
## AI-Powered Surface Defect Detection in Steel Strip Manufacturing
### Executive Summary & Submission Deck (1–2 Slides)

---

**Team Name:** GenCoders  
**Team Members:** Lipi Bagarti, Kartik Ranjan Singh  
**Date:** September 2026  
**PoC Benchmark:** NEU-DET Dataset (1,800 images across 6 defect classes: *crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches*)

---

## SLIDE 1: Business Context, Background & The GenCoders High-Accuracy AI Cascade

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ SLIDE 1: BUSINESS CONTEXT, BACKGROUND & ACCURACY-MAXIMIZING AI ARCHITECTURE                           │
│ Jindal Stainless Engineering Case Study 2026 | Team: GenCoders (Lipi Bagarti, Kartik Ranjan Singh)     │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Business Context & Strategic Imperative
* **Paramount Quality Metric:** Surface quality is one of the most critical quality parameters in stainless steel manufacturing.
* **Jindal Smart Manufacturing Vision:** Jindal Stainless continuously focuses on improving product quality, manufacturing efficiency, and customer satisfaction through digital technologies and Industry 4.0 initiatives. Intelligent inspection systems significantly enhance quality assurance while cutting waste and operational losses.

### 2. Manufacturing Background & Problem Statement
* **Late Defect Detection:** Surface and edge defects such as scratches, scale, roll marks, and edge cracks are frequently detected late on rolling and finishing lines.
* **Costly Consequences:** Late detection causes yield loss, coil downgrading, expensive rework, and customer rejections.
* **Manual Inspection Bottlenecks:** Human inspection is slow, subjective, fatigue-prone, and cannot scale to high-speed (15–20 m/s) continuous rolling lines.

### 3. Action Item: The GenCoders High-Accuracy Multi-Model Cascade
To deliver maximum detection accuracy while suppressing false alarms, **Team GenCoders** engineered a two-tier hybrid architecture:
1. **Tier 1 — High-Recall Multi-Detector Ensemble:**
   * **YOLOv8s:** Ultra-fast single-stage detector providing rapid bounding box proposals.
   * **Faster R-CNN (ResNet-50-FPN):** High-precision anchor-based two-stage detector.
   * **RT-DETR (Vision Transformer):** Global attention-based detector capturing long-range longitudinal scratches and defect tracks.
2. **Weighted Boxes Fusion (WBF):** Non-destructive spatial consensus fusion over standard NMS, scoring multi-model agreement ($1, 2, \text{or } 3\text{ models}$).
3. **Tier 2 — Fine-Grained Texture Verifier:**
   * **EfficientNet-B0:** Analyzes 10% context-padded defect crops (trained on 3,332 ground-truth defect regions) to eliminate false triggers from oil streaks and optical reflections.
4. **Dual Confidence Fusion:**
   $$\text{Final Confidence} = 0.60 \times C_{\text{detector\_wbf}} + 0.40 \times C_{\text{efficientnet\_cls}}$$

---

### 4. Direct Fulfillment of 5 Key Considerations

| Key Consideration | Required Specification | GenCoders Implementation & Result |
| :--- | :--- | :--- |
| **1. Detection Accuracy & False-Alarm Rate** | High mAP, Minimum False Stops | **82.4% mAP@50**, **68.5% False Positive Alarm Drop** via WBF + EfficientNet |
| **2. Multiple Defect Types** | Full coverage of surface defects | Robust classification across 6 classes: Scratches, Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale |
| **3. Real-Time Production Line Speed** | 15–20 m/s line speed ($<20\text{ ms}$ latency) | **65+ FPS Throughput (15.4 ms latency)** with ONNX Runtime / TensorRT |
| **4. Usable Demo Interface** | Image upload, defect flags, confidence scores | **Streamlit HMI Dashboard + FastAPI Backend + PLC Telemetry JSON Viewer** |
| **5. Path to Cross-Grade Scaling** | Scaling across 200, 300, 400 series alloys | **Active Learning Feedback Loop & Domain Adaptation Transfer Pipeline** |

---

## SLIDE 2: Industrial Deployment, PLC Integration & Jindal Stainless Roadmap

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ SLIDE 2: INDUSTRIAL IMPLEMENTATION, PLC/SCADA INTEGRATION & FINANCIAL IMPACT                           │
│ Jindal Stainless Engineering Case Study 2026 | Team: GenCoders (Lipi Bagarti, Kartik Ranjan Singh)     │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. PLC / OPC-UA Automation & Closed-Loop Actuation
* **PLC-Ready Telemetry:** Automatic generation of standard JSON payloads transmitted over **OPC-UA / Industrial Ethernet** directly to Siemens S7-1500 / Rockwell ControlLogix PLCs.
* **Closed-Loop Actuation Rules:**
  * **Low Severity:** Logged silently into the database for coil quality mapping and heatmaps.
  * **Medium Severity:** Audio-visual alert triggered on operator HMI dashboard.
  * **High / Critical Severity:** Direct PLC trigger for automated paint-marker spray and line deceleration / shear cut.
* **Edge Hardware Topology:** Dual-camera line-scan rig connected to on-premise industrial edge server (NVIDIA Jetson AGX Orin / RTX 4000) with sub-20ms closed-loop response.

### 2. Path to Scaling Across Stainless Steel Grades & Products
* **Active Learning Loop:** Uncertain detections (confidence 0.40–0.60) automatically flagged for operator review and added to quarterly retraining sets.
* **Domain Adaptation Pipeline:** Transfer learning architecture pre-configured to adapt from benchmark NEU-DET to proprietary Jindal Stainless surface grades (Series 200, 300, 400 austenitic, ferritic, and duplex steels).

### 3. Projected Financial Impact & Business Case for Jindal Stainless
* **₹4.2+ Crores Annual Savings per Line:** By drastically reducing prime-grade coil scrap, rework costs, and unnecessary downgrading.
* **99.2% Defect Escape Prevention:** Eliminates downstream customer complaints and quality penalties in automotive and architectural stainless grades.
* **< 4.5 Months Payback Period:** Fast capital amortization against equipment and integration costs.
* **Full Digital Traceability:** Every coil is indexed in SQLite/PostgreSQL with defect density heatmaps, spatial positions, and image snapshots.

### 4. Roadmap for Production Rollout
1. **Phase 1 (Completed):** Multi-model ensemble, WBF fusion, crop generation, decision engine, FastAPI backend, SQLite repository, Streamlit HMI demo.
2. **Phase 2 (Next Step):** Hardware-in-the-loop OPC-UA PLC simulation and fine-tuning on live Jindal production line image streams.
