# Case Study Analysis — AI-Powered Surface Defect Detection
### Full written response to Parts 1–30

---

## PART 1 — Understanding the Problem

1. **What is being inspected:** The continuous surface of a moving steel strip (coil
   line) for visible surface-level manufacturing defects.
2. **The steel strip conceptually:** A long, continuously moving flat metal sheet,
   reflective, with a texture/finish that varies by grade — defects appear as local
   deviations in this texture (cracks, embedded particles, scale, pits, scratches).
3. **What the camera captures:** Sequential 2D image frames (or continuous line-scan
   strips) of the surface as it passes under the camera.
4. **What the AI receives:** A preprocessed 2D image (grayscale/normalized) per frame
   or per fixed strip-length segment.
5. **What the AI detects:** Presence, location (bounding region), and type of surface
   defect within each frame.
6. **Classification vs detection vs segmentation:** This problem needs **object
   detection** (localization + classification together), not plain classification.
   Segmentation is a "nice-to-have" upgrade, not a requirement, for a PoC.
7. **Why localization matters:** Downstream systems (PLC, strip marking/rejection)
   need to know *where* on the strip the defect is, not just that "a defect exists
   somewhere in this frame" — classification alone cannot mark/reject a specific
   section.
8. **Why false positives are dangerous:** Every false alarm either triggers an
   unnecessary reject/mark (direct scrap cost) or trains operators to distrust and
   ignore the system (defeats the purpose of automation).
9. **Why false negatives are dangerous:** A missed defect reaches the customer —
   this is the costliest failure mode in quality inspection (rejected shipments,
   reputational damage, contractual penalties).
10. **Why real-time matters:** The strip moves continuously; inspection must keep
    pace with line speed or defective sections pass uninspected, or the line must be
    slowed (production loss).
11. **After detection:** Confidence/severity scoring → decision (log / alert /
    mark / reject) → logged to database → shown on dashboard → optionally sent to
    PLC for physical action.

**Classification vs Detection vs Segmentation — definitions:**
- *Image classification*: one label for the whole image, no location.
- *Object detection*: bounding box + class per defect instance.
- *Semantic segmentation*: pixel-level class map, no instance separation.
- *Instance segmentation*: pixel-level mask per individual defect instance.

**Conclusion:** **Object detection** is the correct task for this case — it gives
location (needed for marking/rejection) and class (needed for severity/reporting)
without the extra annotation/compute cost of pixel-level segmentation, which is not
justified by the case requirements.

---

## PART 2 — Dataset Analysis (NEU-DET)

| Property | Value |
|---|---|
| Size | 1,800 images |
| Classes | 6 (crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches) |
| Images per class | 300 (balanced) |
| Resolution | 200×200, grayscale |
| Annotation format | XML (PASCAL VOC-style bounding boxes) — convertible to YOLO txt |
| Class balance | Balanced (equal per class) — unusual for real production data, where balance is not specified/known |

**Suitability:**
- **YOLO:** Suitable — small image size is fine after resize (YOLO commonly trains at
  640×640; upscaling 200×200 sources is standard practice, though it doesn't add real
  detail).
- **Faster R-CNN:** Suitable technically, but its extra capacity is harder to justify
  on only 1,800 images — higher overfitting risk without strong augmentation.
- **EfficientNet (as classifier only):** Suitable for the classification sub-task on
  cropped patches, but with 300 images/class it needs pretrained ImageNet weights and
  conservative fine-tuning.
- **Production deployment:** **Not sufficient.** NEU-DET is a lab benchmark. Not
  specified in the provided information whether it reflects Jindal's steel grades,
  camera type, lighting, or line speed — must be treated strictly as a PoC dataset.

**Limitations:**
- Small size relative to deep-learning norms.
- Single, fixed image resolution/lighting condition (no variation the way a live
  factory camera would produce over time).
- No metadata on strip speed, camera model, or real defect severity/business
  criticality — these must come from Jindal.

**Domain gap risk:** Real camera resolution, lighting (LED strobe vs. dataset's
studio-like capture), steel grade/finish, and defect morphology at production scale
are all likely different from NEU-DET. This gap must be explicitly called out — it is
the single biggest reason a NEU-DET-only model **cannot** be presented as
production-ready.

---

## PART 3 — Data Split Strategy

**Proposed 70/15/15 split is reasonable as a starting point, with caveats:**

- **Leakage risk:** If any images are near-duplicates (cropped from the same original
  defect photo, or augmented versions of one source image), random splitting can leak
  the same defect instance into both train and test, inflating measured accuracy.
  NEU-DET's images are curated and largely distinct, but this should still be
  explicitly checked (e.g., perceptual hashing to catch duplicates) before finalizing
  a split.
- **Class balance:** Split must be **stratified per class** (not purely random) so
  each of the 6 classes is represented proportionally in train/val/test.
- **Augmentation timing:** Augmentation must be applied **only to the training set,
  after splitting** — never before splitting (this would leak augmented copies of a
  test image into training) and never to validation/test sets (these must reflect
  real, unaltered evaluation conditions).
- **Test set must remain untouched:** No augmentation, no preprocessing tuning based
  on test performance — it is used exactly once, at the end, to report final metrics.

**Recommended final split:** 70% train / 15% val / 15% test, **stratified by class**,
with an explicit duplicate/near-duplicate check before splitting.

---

## PART 4 — Data Preprocessing

| Operation | Why useful | Downside | Apply during |
|---|---|---|---|
| Resize | Required — models need fixed input size | Slight detail loss/distortion if aspect ratio not preserved | Train + Inference |
| Grayscale | Steel surface defects are texture/luminance-based, not color-based; reduces input complexity | Loses any color-based defect cues (rare for steel, but not verified for Jindal) | Train + Inference |
| Normalization | Standard requirement for stable neural network training | None significant | Train + Inference |
| CLAHE | Enhances subtle low-contrast defects (crazing, fine scratches) against reflective background | Can amplify sensor noise/glare into false texture if overdone | Train + Inference (same setting both places) |
| Gaussian/median denoise | Removes sensor noise, stabilizes CLAHE output | **Risk:** aggressive denoising can blur/remove exactly the fine texture that IS the defect (crazing, hairline scratches) — must be light | Train + Inference, light strength only |
| Brightness/contrast variation | Simulates real lighting inconsistency on a reflective line | none if kept realistic | Train only (augmentation) |
| Rotation | Steel strip defects have no fixed orientation relative to camera | Large rotations are unrealistic for a fixed camera geometry — keep small (±10–15°) | Train only |
| Horizontal flip | Valid — strip has no inherent left/right defect bias | none | Train only |
| Vertical flip | Valid only if the camera setup genuinely has no fixed "up" reference for defects (strip direction) — verify against real camera geometry before using | Could be physically unrealistic if strip has a directional characteristic | Train only, verify first |
| Random crop | Can help generalization | **Risk:** may crop out the defect entirely, especially on small 200×200 images — use conservative crop ratios | Train only, light |
| Blur (as augmentation) | Simulates minor motion blur/focus variation from a moving strip | Risk of hiding fine defects if overused | Train only, light, low probability |

**Key principle enforced:** every preprocessing/augmentation choice is evaluated
against the specific risk that steel-defect texture is *subtle* — anything that
smooths or blurs the image too aggressively can literally remove the signal the model
needs to detect (especially for crazing and fine scratches). Preprocessing must stay
conservative and be validated empirically (Part 22, ablation study), not applied
"because it's common."

---

## PART 5 — Data Augmentation Strategy (recommended ranges)

| Augmentation | Probability | Range |
|---|---|---|
| Horizontal flip | 0.5 | — |
| Vertical flip | 0.3 (verify against real geometry first) | — |
| Rotation | 0.5 | ±10–15° |
| Brightness | 0.4 | ±15% |
| Contrast | 0.4 | ±15% |
| Gaussian noise | 0.2 | low sigma (simulate sensor noise only) |
| Light blur | 0.15 | small kernel only |
| Random crop | 0.3 | crop ≥ 80% of original area, defect-aware if possible |
| Scale jitter | 0.3 | ±10% |
| Perspective transform | **Not recommended** | Real camera geometry for a fixed overhead/line-scan setup is essentially planar; perspective warping does not reflect real imaging conditions and can distort defect shape unrealistically |

**Explicitly avoided:** aggressive elastic transforms, heavy blur, large rotations,
color-jitter (image is grayscale/texture-based) — these risk changing the physical
appearance of a defect into something the real camera would never produce.

---

## PART 6 — Model Comparison

| Model | Purpose | Det./Class. | Accuracy potential | Speed | Compute cost | Dataset (1.8k imgs) fit | Deployment fit | Complexity | Main advantage | Main disadvantage |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n | Detect+classify | Both | Medium | Fastest | Very low | Good | Excellent (edge) | Low | Extremely fast, tiny footprint | Lower accuracy on subtle/small defects |
| YOLOv8s | Detect+classify | Both | Medium-high | Very fast | Low | Good | Excellent | Low | Best speed/accuracy balance for small models | Still limited on very subtle textures |
| YOLOv8m | Detect+classify | Both | High | Fast | Medium | Good | Good (needs decent GPU/edge module) | Medium | Strong accuracy while still real-time | Heavier than s/n, needs benchmarking on target device |
| Faster R-CNN (R50) | Detect+classify | Both | High (esp. small objects) | Slow (~5–15 FPS typical) | High | Workable but higher overfit risk on 1.8k imgs | Poor for strict real-time edge use | High | Strong localization accuracy | Too slow for continuous line-speed inspection without heavy hardware |
| RT-DETR | Detect+classify | Both | High | Medium-fast | Medium-high | Workable, benefits from more data | Possible on strong edge GPU | High | Good global context, avoids NMS issues | Newer/less battle-tested, added engineering complexity |
| EfficientNet-B0 | Classify only | Classification only | High (as classifier) | Fast | Low | Good (as a classifier on crops) | Good | Low | Strong fine-grained classification accuracy | Cannot localize — must be paired with a detector, adds a pipeline stage |

**Selection:** **YOLOv8s or YOLOv8m** as the primary model — it is the only option in
this table that natively does both detection and classification in one real-time-fast
pass, fits the dataset size without excessive overfit risk, and is realistically
deployable on edge hardware. This is **not** the most complex option in the table by
design — Faster R-CNN and RT-DETR are documented as alternatives for a
maximum-accuracy/offline configuration only (see Part 7).

---

## PART 7 — Questioning the Proposed Ensemble

**Original proposal:** YOLOv8m + Faster R-CNN → WBF → EfficientNet-B0 → RT-DETR
(if uncertain).

**Critical evaluation:**

1. **Is it over-engineered?** Yes, for this dataset size and real-time constraint.
   Four models chained together is justified when a single model has a *measured*
   accuracy ceiling that ensembling is proven (experimentally) to break — not as a
   default starting design.
2. **Will it actually improve accuracy?** Possibly marginally, but this is
   **unverified** without running Part 21's experiments — the case study rules
   explicitly require evidence, not assumption.
3. **Will it hurt real-time performance?** Yes, significantly. Running YOLOv8m +
   Faster R-CNN in parallel, fusing boxes, cropping, and running a second
   classifier per detection multiplies latency several times over. This directly
   conflicts with the case's real-time requirement.
4. **Is WBF appropriate here?** WBF is designed to fuse predictions from
   *different model families* in competitions (e.g., Kaggle) where inference time is
   not constrained. In a real-time industrial line, its latency cost is hard to
   justify unless the accuracy gain is proven necessary.
5. **Does EfficientNet add value after YOLO already classifies?** Only if YOLO's
   *own* classification head is measurably weak on specific confusable classes
   (e.g., crazing vs. rolled-in scale) — this must be checked from YOLO's own
   confusion matrix first, before adding a second model.
6. **Does RT-DETR justify its added latency?** Not by default — it's a reasonable
   tie-breaker for a small number of low-confidence cases sent to human review, not
   as a real-time third detector on every frame.
7. **Can this run on an industrial edge device?** Realistically no, not while meeting
   strict line-speed real-time constraints — this stack is better suited to a
   GPU server doing offline/batch quality audits, not inline inspection.
8. **Is the complexity justified for a competition PoC?** No — judges evaluating
   "feasibility" and "technical robustness" are likely to see an unjustified 4-model
   real-time ensemble as a red flag, not a strength, unless backed by ablation
   evidence (Part 22) that a simpler model genuinely underperforms.

### Recommended Architectures

**Architecture A — Best balance (PRIMARY RECOMMENDATION):**
- Models: YOLOv8s or YOLOv8m only.
- Data flow: Camera → preprocess → YOLOv8 → confidence/size/temporal filters →
  severity → action.
- Expected benefit: Real-time speed, single model to maintain/explain/deploy,
  fits dataset size, still gives class + confidence + box.
- Main limitation: Slightly lower ceiling on very subtle/small defects compared to
  a two-stage or ensemble approach.
- When to use: Default choice for the real-time production line and for this
  competition's live demo.

**Architecture B — Maximum accuracy (offline/audit mode):**
- Models: YOLOv8m (or Faster R-CNN) + EfficientNet-B0 second-stage classifier on
  crops.
- Data flow: Detector proposes regions → crop → EfficientNet gives final
  fine-grained class.
- Expected benefit: Higher classification accuracy on visually similar defect
  classes.
- Main limitation: Added latency (two forward passes) — not suited to strict
  real-time line speed, better for periodic/batch quality audits or lower-speed
  lines.
- When to use: When Part 21 experiments show YOLO's own classification head has a
  measurable weakness the second stage fixes.

**Architecture C — Maximum real-time speed:**
- Models: YOLOv8n only, exported to TensorRT INT8.
- Expected benefit: Highest possible FPS, lowest compute cost, best fit for
  resource-constrained edge modules.
- Main limitation: Lowest accuracy ceiling of the three options.
- When to use: Very high line speed or very constrained edge hardware where A is
  not fast enough after benchmarking.

**Final single recommendation: Architecture A (YOLOv8s/m single-stage).** This is
what the codebase in this repo implements by default, with Architecture B's
second-stage classifier included as an optional, switchable module
(`training/train_classifier.py`) rather than a hard dependency.

---

## PART 8 — Detection + Classification Strategy

**Approach 1 (YOLO does both directly):** Simpler pipeline, lower latency, one model
to train/maintain/deploy. Risk: classification accuracy tied to detector's own head,
which is optimized jointly with localization loss (a compromise, not classification-
only optimal).

**Approach 2 (YOLO detects → crop → EfficientNet classifies):** Higher potential
classification accuracy since the second model is optimized purely for
classification on a tighter, already-localized crop. Cost: extra latency, extra model
to maintain, extra failure point (bad crop → bad classification).

**Recommendation:** Start with **Approach 1** for the PoC and the real-time line
(Architecture A). Only add Approach 2 as an optional second stage if YOLO's own
per-class confusion matrix (from Part 21 experiments) shows a specific, measurable
weakness that a dedicated classifier fixes — implemented but not required by default
in this codebase.

---

## PART 9 — Training Strategy

1. **Transfer learning:** Start from COCO-pretrained YOLOv8 weights (Ultralytics
   default) — essential given the small dataset.
2. **Pretrained weights:** Yes, mandatory — training from scratch on 1,800 images
   will underfit/overfit badly.
3. **Freeze/unfreeze:** Initially freeze backbone for a few warmup epochs, then
   unfreeze fully for fine-tuning (standard Ultralytics behavior with a short
   freeze schedule).
4. **Learning rate:** Start ~1e-3 with cosine or linear decay (Ultralytics default
   scheduler); lower to ~1e-4 if loss is unstable.
5. **Optimizer:** SGD or AdamW (Ultralytics default AdamW works well for small
   datasets).
6. **Batch size:** As large as GPU memory allows for stable batch-norm statistics
   (e.g., 16–32 for a single mid-range GPU); reduce if memory-constrained.
7. **Epochs:** 100–200 with early stopping — small datasets converge fast and
   overfit if trained too long.
8. **Early stopping:** Monitor validation mAP@0.5, stop if no improvement for
   ~20–30 epochs (patience).
9. **LR scheduling:** Cosine decay (Ultralytics default) works well.
10. **Weight decay:** Standard small value (Ultralytics default, ~5e-4) to reduce
    overfitting.
11. **Augmentation:** As specified in Part 5.
12. **Class imbalance:** NEU-DET itself is balanced; if real Jindal data later
    introduces imbalance, use class-weighted loss or oversampling of rare classes.
13. **Hyperparameter tuning:** Use Ultralytics' built-in tuning utilities or a small
    grid/Bayesian search over LR, batch size, and augmentation strength — validated
    on the validation set only, never the test set.

**Overfitting mitigation (small dataset):** pretrained weights + strong-but-realistic
augmentation + early stopping + weight decay + dropout in the classifier head (if
Approach 2 is used) + strict train/val/test separation.

---

## PART 10 — Detection Metrics

- **Precision** = TP / (TP + FP) — of all predicted defects, how many were real.
- **Recall** = TP / (TP + FN) — of all real defects, how many were caught.
- **F1** = harmonic mean of precision and recall.
- **IoU** = overlap ratio between predicted and ground-truth box; threshold (e.g.,
  0.5) determines if a detection counts as a match.
- **mAP@0.5** = mean average precision at IoU threshold 0.5 across all classes —
  standard detection benchmark metric.
- **mAP@0.5:0.95** = mAP averaged over IoU thresholds 0.5 to 0.95 — a stricter,
  localization-sensitive metric.
- **False Positive Rate** = false alarms relative to total predictions/frames.
- **False Negative Rate** = missed real defects relative to total real defects.

**Why recall and false-alarm rate matter more than raw accuracy here:** "Accuracy"
on an imbalanced real-world stream (mostly non-defective frames) is a misleading
metric — a model that predicts "no defect" almost always would still show high
accuracy while missing every real defect. For industrial inspection, **missing a
real defect (low recall) risks shipping a defective product**, and **too many false
alarms (high FPR) erodes operator trust and increases unnecessary rejects**. Both
directly map to business cost in a way plain accuracy does not.

---

## PART 11 — Confidence Threshold Optimization

**Method:** Sweep thresholds (0.10 → 0.90 in steps of 0.10, refined further around
the best region) on the **validation set only**. For each threshold, compute
precision, recall, F1, false positives, false negatives. Plot a precision-recall
curve and select the threshold based on the industrial cost trade-off, not simply
the maximum F1 point:

- If missing a defect (false negative) is far costlier than a false alarm → bias
  threshold lower (favor recall).
- If false alarms cause expensive unnecessary line stoppages → bias threshold higher
  (favor precision).

**Per-class thresholds:** Different defect types can have different business
severity (e.g., a deep crack-type defect vs. a minor surface patch) — thresholds can
and should be tuned independently per class once real severity/cost data from Jindal
is available, rather than using one global threshold for all 6 classes.

---

## PART 12 — False Alarm Reduction (Decision Layer)

```
YOLO detection
   ↓
Confidence filter        (reject detections below per-class tuned threshold)
   ↓
Size filter               (reject boxes below a minimum pixel-area — sensor noise / artifacts rarely form large coherent regions)
   ↓
Spatial/ROI filter         (ignore detections outside the actual strip region, e.g. edge/background artifacts)
   ↓
Temporal consistency       (require the same defect region to be seen across N consecutive/overlapping frames)
   ↓
Severity classification    (map type+size+persistence to a severity level)
   ↓
Final alert
```

Each stage removes a different class of false positive: confidence filtering removes
low-certainty noise, size filtering removes tiny sensor artifacts, ROI filtering
removes non-strip background detections, and temporal confirmation removes one-off
transient false positives (lighting flicker, reflection glints) that don't persist
across frames the way a real physical defect does. This staged design is implemented
in `postprocessing/decision_engine.py`.

---

## PART 13 — Multi-Frame / Temporal Logic

Since a defect physically persists as the strip moves, it should appear in more than
one captured frame if the camera's field-of-view overlaps between consecutive
captures (or between overlapping crops of a continuous line-scan feed).

**Design:**
- Use **encoder-based triggering** (capture tied to physical strip movement, not
  wall-clock time) so frame overlap is consistent regardless of line-speed changes.
- Maintain a short-lived spatial tracker (simple IoU-based matching across
  consecutive frames, e.g. a lightweight tracker such as a Kalman-filter/IoU tracker)
  that links detections of the same physical defect across frames using the known
  strip displacement between captures.
- **Confirmation rule:** a defect is confirmed only if matched detections persist
  across a minimum number of consecutive/overlapping frames (exact frame count
  depends on camera overlap percentage and strip speed — **not specified in the
  provided information**, must be derived from real line parameters).
- **Avoiding missed brief defects:** set the confirmation window based on the
  *minimum physical defect length* that matters for quality (a business/engineering
  input from Jindal), not an arbitrary frame count — if a defect is smaller than one
  frame's overlap, single-frame detection with a slightly higher confidence
  threshold should still flag it rather than requiring multi-frame confirmation.

No specific line speed or frame rate is assumed here since none was provided —
`configs/config.yaml` exposes these as configurable parameters to be filled in with
real Jindal line data.

---

## PART 14 — Severity and Action Logic

Severity should **not** be arbitrarily assigned by the AI team — it should be derived
from a combination of:

- **Defect type** (some types are inherently more critical for a given product/use
  case per metallurgical standards).
- **Defect area/length** (larger defects generally more severe).
- **Defect density** (multiple defects in one strip section compound severity).
- **Position** (edge vs. center of strip may have different tolerance per customer
  spec).
- **Customer/quality specification** (external requirement, not something the model
  invents).

**Framework (illustrative structure only — actual thresholds must come from Jindal's
QA/metallurgy team):**

| Level | Trigger pattern | Action |
|---|---|---|
| Low | Small area, low-severity type, single frame | Log only |
| Medium | Moderate area or medium-severity type, confirmed across frames | Operator alert on dashboard |
| High | Large area or high-severity type, confirmed | Mark/flag defective section, PLC signal |
| Critical | Very large / dense / high-risk type | Immediate escalation, possible line stop per Jindal protocol |

This mapping is implemented as a configurable table in `configs/config.yaml` —
**not hardcoded business logic** — so real severity/action rules can be inserted
without changing code.

---

## PART 15 — Real-Time Deployment Architecture

```
Camera → Frame acquisition → Preprocessing → AI inference → Post-processing
       → Decision engine → PLC/alert → Database → Dashboard
```

- **Edge deployment (Jetson-class device):** Good for low-latency, no network
  dependency, physically near the camera — appropriate default for inline
  inspection.
- **Industrial PC with GPU:** More compute headroom, easier to service/upgrade,
  appropriate if edge module compute is insufficient.
- **ONNX:** Framework-agnostic export format, intermediate step before TensorRT.
- **TensorRT:** NVIDIA-specific runtime optimization — reduces latency significantly
  on NVIDIA edge/GPU hardware, the standard choice given Jetson/industrial-GPU
  deployment.
- **FP16:** Good accuracy/speed trade-off, widely supported, first optimization to
  try.
- **INT8:** Faster still, but requires a calibration step and can lose accuracy on
  subtle defects — use only if FP16 doesn't meet throughput and validate accuracy
  drop carefully before deploying.

**Required throughput — how to calculate it (no FPS invented):**

```
Required FPS ≥ (Strip speed [m/min] × 1000 / 60) / (Field-of-view length per frame [mm] × (1 − overlap fraction))
```

All of strip speed, field-of-view, and camera overlap are **not specified in the
provided information** — `configs/config.yaml` exposes these as parameters to be
filled with real line data; only then can a target FPS be defined and the model
variant (n/s/m) and optimization level (FP16/INT8) be chosen to meet it, followed by
actual on-device benchmarking.

---

## PART 16 — Camera and Lighting Design

**Camera:**
- **Line-scan:** Preferred for a continuously moving strip at production line speed
  — captures one line of pixels per trigger, naturally suited to continuous motion
  without motion blur, and stitches into a continuous image.
- **Area-scan:** Acceptable and simpler for a **prototype/PoC** (e.g., using static
  or slow-moving sample images, or a benchtop demo) where full line-speed capture
  isn't required yet.
- Resolution and frame rate: must be selected once real strip speed and required
  defect-detection minimum size are known — **not specified in the provided
  information**, treated as a design assumption to validate with Jindal.

**Lighting:**
- Diffused/controlled LED illumination to minimize glare from the reflective steel
  surface (direct/specular lighting would create bright spots that look like or hide
  defects).
- Stable, consistent illumination is critical — the model is trained on one lighting
  condition (NEU-DET's) and any real deployment lighting must be as consistent as
  possible to reduce the domain gap.

**Synchronization:** Encoder-based triggering ties image capture to actual physical
strip movement rather than wall-clock time — this keeps frame overlap consistent
even if line speed varies, which is what enables reliable temporal confirmation
(Part 13).

All specific numeric camera/lighting specs above are **explicitly framed as design
assumptions**, not confirmed Jindal specifications.

---

## PART 17 — Industrial Integration

**Protocols:** OPC-UA is the more modern, secure, and widely supported industrial
standard for this kind of integration; Modbus is simpler/older and may be required
if that's what the existing PLC infrastructure uses. The correct choice depends on
Jindal's existing PLC/automation vendor — **not specified in the provided
information**.

**Payload sent to PLC (example structure, implemented in
`postprocessing/decision_engine.py`):**

```json
{
  "defect_type": "scratches",
  "confidence": 0.91,
  "severity": "medium",
  "strip_position": "encoder_ref_or_meters",
  "timestamp": "ISO8601",
  "action": "alert"
}
```

**Physical marking/rejection:** Typically done via a PLC-triggered mechanical
marker/sprayer at a known downstream distance from the camera (using strip speed and
encoder position to time the action correctly), or automatic diversion at a rejection
station — exact mechanism depends on Jindal's existing line equipment.

---

## PART 18 — Database and Dashboard

**Dashboard shows:** live feed / last captured frame, bounding boxes, defect class,
confidence, timestamp, strip position, severity, daily defect count, defect-type
distribution, and (once operators start verifying) a false-alarm feedback rate.

**Database schema (implemented conceptually in `postprocessing/decision_engine.py`
output format, ready to insert into PostgreSQL/MySQL):**

```
Defect_ID        (PK)
Timestamp
Strip_ID
Position
Defect_Type
Confidence
Severity
Image_Path
Operator_Verified   (nullable, filled in after human review)
Final_Label          (nullable, corrected label if operator disagrees)
```

The `Operator_Verified` / `Final_Label` fields exist specifically to support the
continuous-learning feedback loop (Part 19).

---

## PART 19 — Continuous Learning

```
Production → Predictions → Operator verification → False positives/negatives
→ New labeled data → Dataset versioning → Retraining → Validation → Deployment
```

**Model drift triggers that require monitoring/retraining:**
- New steel grades introduced (different surface texture/reflectivity).
- Lighting changes (bulb aging, seasonal factors, maintenance changes).
- Camera aging/recalibration.
- Surface finish process changes.
- New/unseen defect types appearing that weren't in the original 6 classes.

**Monitoring after deployment:** Track live confidence distributions, false-alarm
feedback rate from operators, and per-class detection rate over time; a sustained
shift in any of these is the practical signal that retraining/domain-adaptation is
due — not a fixed calendar schedule alone.

---

## PART 20 — Domain Gap: NEU-DET vs. Real Jindal Production

NEU-DET is unlikely to fully represent Jindal's real conditions because of (all
explicitly **not specified in the provided information**, so listed as known
unknowns):
- Different steel grades and surface finishes than the benchmark's source material.
- Different industrial camera sensor/resolution/color response.
- Different factory lighting setup than the dataset's capture conditions.
- Real high-speed strip movement (motion blur, encoder-triggered capture) vs. the
  dataset's static captured images.
- Possibly different defect morphology at Jindal's specific process conditions.

**Domain adaptation plan (implemented as project phases, not code, since real data
isn't available yet):**

```
Phase 1: Train on NEU-DET (this repo)
Phase 2: Collect real Jindal-line images (camera installed, no labels yet)
Phase 3: Label images (with Jindal QA input on true defect types/severity)
Phase 4: Fine-tune the NEU-DET-pretrained model on real labeled data
Phase 5: Validate on a held-out real-data test set
Phase 6: Deploy
Phase 7: Continuously retrain via the Part 19 feedback loop
```

This phased structure is the core reason the codebase separates
`training/train_yolo.py` (generic, dataset-path-driven) from the NEU-DET-specific
conversion script — swapping in real Jindal data later requires no code changes,
only a new `configs/data.yaml`.

---

## PART 21 — Experimental Design

| Experiment | What it tests |
|---|---|
| 1. Baseline YOLOv8 (no augmentation, default preprocessing) | Floor performance |
| 2. + CLAHE/preprocessing | Does contrast enhancement help subtle-defect detection? |
| 3. + full augmentation (Part 5) | Does it improve generalization/reduce overfitting? |
| 4. + decision engine (confidence/size/temporal filters) | Does false-alarm rate drop without hurting recall? |
| 5. + second detector (Faster R-CNN) fused | Does ensembling measurably raise mAP enough to justify latency cost? |
| 6. + EfficientNet second-stage classifier | Does it fix specific class-confusion pairs seen in Experiment 1's confusion matrix? |

**Metrics tracked per experiment:** mAP@0.5, mAP@0.5:0.95, precision, recall, F1,
false positives, false negatives, inference latency (ms), FPS, GPU memory.

**Rule enforced:** the final architecture choice (Part 7) is justified by *running
Experiments 1–4 first*; Experiments 5–6 (ensemble/second-stage) are only adopted if
they show a measurable gain over Experiment 4 that's worth their added latency —
this repo defaults to the Experiment 1–4 pipeline and includes optional code
(`training/train_classifier.py`) to run Experiment 6 if justified.

---

## PART 22 — Ablation Study

```
Baseline
  + Augmentation           → measure Δ mAP, Δ overfitting gap (train vs val)
    + CLAHE                → measure Δ recall on subtle classes (crazing, rolled-in_scale)
      + Confidence optimization → measure Δ false-positive rate at fixed recall
        + Temporal filtering     → measure Δ false-positive rate on a simulated multi-frame stream
          + Second-stage classifier → measure Δ per-class F1 vs added latency
            + Full ensemble          → measure Δ mAP vs added latency (likely diminishing returns)
```

**Decision rule:** if a component's accuracy/false-alarm improvement is small
relative to its added latency/complexity cost, it is documented as **not
recommended for the real-time pipeline**, matching the Part 7 conclusion — this is
how the ensemble was ruled out analytically ahead of running the full experiment,
and how the experiments would confirm or correct that call.

---

## PART 23 — Business Impact Framework

**Formulas (no invented Jindal numbers — variables to be filled with real data):**

```
Scrap savings      = Baseline_scrap_cost − Post_AI_scrap_cost
Rework savings     = Baseline_rework_cost − Post_AI_rework_cost
Rejection savings  = Avoided_customer_rejection_cost
Annual_Benefit     = Scrap_savings + Rework_savings + Rejection_savings
                     + (Reduced_manual_inspection_labor_cost)

ROI = (Annual_Benefit − Annual_System_Cost) / Annual_System_Cost
```

**Qualitative business value (not quantified without real data):** earlier defect
detection (catch issues before more processing is invested in a defective section),
reduced downgrading of product grade, improved consistency vs. manual/human visual
inspection (fatigue-independent), better traceability (every defect logged with
position/time/image), faster corrective action on upstream process issues (repeated
defect types/locations can point to a root-cause process problem).

---

## PART 24 — Scalability

Modular layers, each independently upgradeable:

```
Camera Layer → Preprocessing Layer → AI Model Layer → Decision Layer
             → Integration Layer → Analytics Layer
```

- **New steel grades/finishes:** handled via domain-adaptation fine-tuning (Part 20)
  without changing the pipeline architecture.
- **New production lines:** each line gets its own camera+config instance; the
  Model/Decision/Integration/Analytics layers are shared/reusable.
- **New defect types:** requires new labeled data and either retraining the existing
  model (adding a class) or, if very different in nature, potentially a
  supplementary model — the modular design isolates this change to the Model Layer.
- **Different line speeds:** handled by the throughput formula in Part 15 (choose
  model variant + optimization level accordingly) without pipeline redesign.

---

## PART 25 — Competition Demo

**Recommended tool: Streamlit.** Simplest reliable option for a live judged
presentation — single Python file, no separate frontend/backend to keep in sync
during a demo (lower live-demo failure risk than a React+FastAPI split), while still
looking polished. Gradio is a close second choice; Flask/React+FastAPI add
unnecessary complexity for a demo that just needs "upload image → see result."

**Demo flow (implemented in `dashboard/app.py`):**
1. User uploads a steel strip image.
2. Image is preprocessed and run through the trained YOLOv8 model.
3. Decision engine applies confidence/size filtering and severity mapping.
4. Output: annotated image with bounding boxes + labels, a results table (defect
   type, confidence, severity), and a final overall decision (e.g., "⚠️ DEFECTIVE" /
   "✅ CLEAN").

---

## PART 26 — Failure Cases

| Failure case | Detection | Mitigation |
|---|---|---|
| Very small defects | Low recall on small boxes in validation metrics | Higher-resolution capture, smaller anchor/scale tuning, or dedicated small-object augmentation |
| Low contrast | Confusion concentrated in specific classes (crazing) | CLAHE preprocessing, validate its effect via ablation (Part 22) |
| Excessive glare | Spurious false positives correlated with bright regions | Better diffused lighting design (Part 16), glare-aware preprocessing |
| Motion blur | Accuracy drop correlated with line-speed spikes | Line-scan camera + encoder trigger reduces this; validate at real speed |
| New defect type | Unknown class shows as low-confidence "closest match" or missed entirely | Human review queue for low-confidence detections; feeds Part 19 retraining loop |
| Overlapping defects | Merged/incorrect boxes | NMS tuning, evaluate under Experiment 5 (second detector) if this is a frequent real case |
| Unseen steel grade | Sudden accuracy drop on that grade | Domain-adaptation fine-tuning (Part 20) |
| Different surface finish | Similar to above | Same mitigation, plus per-grade model variants if needed at scale |
| Camera failure | Frame stream stops/corrupts | Health-check/watchdog on camera feed with alert to maintenance |
| Lighting failure | Sudden shift in prediction confidence distribution | Live monitoring (Part 19) should flag this as an anomaly, not silently degrade |
| Model uncertainty | Confidence near threshold | Route to human-review queue rather than force a decision |
| False positives / negatives | Ongoing operator feedback | Logged via `Operator_Verified`/`Final_Label` fields, feeds retraining |

---

## PART 27 — Final Recommended Architecture

```
CAMERA                → Input: physical strip | Output: raw frame | Tech: line-scan (prod) / area-scan (PoC) camera | Purpose: image acquisition
IMAGE ACQUISITION      → Input: raw sensor data | Output: digital frame | Tech: encoder-triggered capture | Purpose: consistent, motion-synced capture
PREPROCESSING          → Input: raw frame | Output: normalized/enhanced frame | Tech: OpenCV (grayscale, CLAHE, resize, light denoise) | Purpose: enhance defect signal, standardize input
PRIMARY AI MODEL        → Input: preprocessed frame | Output: boxes + class + confidence | Tech: YOLOv8s/m (Ultralytics) | Purpose: real-time detection + classification
OPTIONAL 2ND-STAGE MODEL → Input: cropped detection | Output: refined class | Tech: EfficientNet-B0 | Purpose: only if experiments justify it (Part 21)
POST-PROCESSING         → Input: raw detections | Output: filtered detections | Tech: confidence/size/ROI filters | Purpose: remove low-quality/noise detections
CONFIDENCE + SEVERITY    → Input: filtered detections | Output: severity-tagged detections | Tech: rule-based mapping (config-driven) | Purpose: translate AI output into business-relevant severity
TEMPORAL VALIDATION      → Input: per-frame detections over time | Output: confirmed defects | Tech: IoU tracker across encoder-synced frames | Purpose: eliminate one-frame false positives
FINAL DECISION           → Input: confirmed, severity-tagged defect | Output: action | Tech: rule engine | Purpose: decide log/alert/mark/reject
ALERT / PLC / DATABASE   → Input: final decision | Output: signal + record | Tech: OPC-UA/Modbus + SQL DB | Purpose: physical action + persistent record
DASHBOARD                → Input: database | Output: visual monitoring | Tech: Streamlit/Grafana | Purpose: human oversight, trend monitoring, feedback capture
```

---

## PART 28 — Implementation Roadmap

| Phase | Tasks | Tools | Expected output | Success criteria |
|---|---|---|---|---|
| 1. Dataset prep | Download NEU-DET, convert annotations, split, verify no leakage | Python, this repo's `preprocessing/` scripts | YOLO-format dataset | Correct class counts per split |
| 2. Baseline model | Train YOLOv8s with default settings | `training/train_yolo.py` | Baseline mAP | Reasonable mAP@0.5 on val set |
| 3. Improved model | Add CLAHE + augmentation, tune confidence threshold | Same script, updated config | Improved mAP, tuned threshold | Measurable improvement over baseline |
| 4. Validation | Run Part 21 experiments, ablation (Part 22) | Experiment tracking (e.g. simple CSV/MLflow) | Comparison table | Evidence-based architecture choice confirmed |
| 5. Decision engine | Implement confidence/size/temporal/severity logic | `postprocessing/decision_engine.py` | Filtered, severity-tagged output + PLC-ready JSON | False-positive rate reduced without recall collapse |
| 6. Demo | Build Streamlit app | `dashboard/app.py` | Working upload → result demo | Runs reliably end-to-end on sample images |
| 7. Deployment simulation | Export ONNX, discuss TensorRT/edge benchmarking plan | `deployment/export_onnx.py` | Exported model + benchmarking plan | Documented required-FPS calculation (Part 15) |
| 8. Industrial scaling | Document PLC integration, database schema, domain-adaptation plan | Docs (Parts 17–20) | Full deployment/scaling plan | Clear, evidence-based path from PoC to production |

---

## PART 29 — Code Structure

```
steel_defect_detection/
├── data/               # raw NEU-DET + converted YOLO-format dataset
├── configs/            # data.yaml (classes/paths), config.yaml (thresholds, severity rules)
├── preprocessing/       # annotation conversion, CLAHE/denoise/resize pipeline
├── training/            # YOLOv8 training script, optional EfficientNet second-stage
├── inference/            # single-image/single-frame inference wrapper
├── postprocessing/       # decision engine: filters, temporal logic, severity, PLC JSON
├── dashboard/             # Streamlit competition demo
├── deployment/            # ONNX export (TensorRT conversion documented)
├── tests/                 # unit tests for decision engine logic
├── docs/                  # this file — full written case study analysis
├── requirements.txt
└── README.md
```

Each folder maps 1:1 to a stage in the Part 27 architecture diagram, so the codebase
structure itself documents the pipeline.

---

## PART 30 — JUDGE-READY FINAL PROPOSAL

**1. Problem Understanding:** Real-time object detection (not plain classification)
of 6 steel surface defect types on a moving strip, where location matters for
downstream marking/rejection and speed matters to keep pace with the line.

**2. Proposed Solution:** A single-stage YOLOv8 detector pipeline with a rule-based
decision layer (confidence + size + temporal filtering + severity mapping),
deliberately chosen over a heavier multi-model ensemble after critical evaluation
(Part 7) — the ensemble is documented as an optional accuracy-maximizing extension,
not the default.

**3. System Architecture:** See Part 27 diagram.

**4. Dataset:** NEU-DET (1,800 images, 6 balanced classes) — explicitly scoped as a
PoC dataset, with a defined domain-adaptation path (Part 20) to real Jindal
production data.

**5. AI Methodology:** Transfer learning from COCO-pretrained YOLOv8, fine-tuned on
NEU-DET with conservative, defect-aware augmentation (Part 5).

**6. Training Strategy:** See Part 9 — pretrained weights, early stopping, weight
decay, class-balanced data, validated hyperparameters.

**7. Validation Strategy:** Stratified 70/15/15 split with leakage checks (Part 3),
evaluated on mAP, precision, recall, F1 (Part 10), with confidence threshold tuned on
validation data per class (Part 11).

**8. False Alarm Reduction:** Staged decision engine — confidence, size, ROI,
temporal confirmation (Parts 12–13).

**9. Real-Time Deployment:** ONNX/TensorRT export, throughput sized to actual line
speed via an explicit formula (Part 15), not an invented FPS number.

**10. Industrial Integration:** OPC-UA/Modbus JSON payload to PLC with defect type,
confidence, severity, position, timestamp (Part 17).

**11. Dashboard:** Live monitoring with defect feed, trends, and an operator
feedback mechanism that feeds continuous retraining (Parts 18–19).

**12. Scalability:** Modular layered architecture supporting new grades, lines,
defect types, and speeds without redesign (Part 24).

**13. Innovation:** Evidence-driven architecture selection (ablation-justified,
not assumption-driven), config-driven severity/business rules (not hardcoded),
temporal/encoder-synced false-alarm reduction tailored to a continuously moving
strip.

**14. Business Impact:** Quantification framework provided (Part 23) — ready for
real Jindal cost data, not invented numbers.

**15. Limitations:** PoC trained on a public benchmark with an acknowledged domain
gap from real production imaging (Part 20); real camera specs, line speed, and true
defect severity/cost data are not yet available and are required before claiming
production readiness.

**16. Future Scope:** Real-data fine-tuning, per-class threshold tuning with real
severity/cost input, expansion to new defect types/steel grades, and evaluation of
Architecture B (second-stage classifier) or ensemble methods **if** experiments show
they're justified by accuracy gains that outweigh their latency cost.
