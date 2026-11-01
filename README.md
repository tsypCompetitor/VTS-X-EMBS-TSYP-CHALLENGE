# CABINGUARD-ADI

**A Zero-Compliance Multimodal In-Cabin Framework for Prodromal Prediction and Fail-Safe Actuation in Acute Driver Incapacitation**

*Research Prototype — TSYP14 Technical Challenge*

---

> [!WARNING]
> **RESEARCH PROTOTYPE DISCLAIMER — NO MEDICAL OR PRODUCTION VEHICLE CLAIMS**
> This repository contains research prototyping, simulation, and benchmarking code. It is **not** a certified medical diagnostic device and does **not** interface with physical vehicle CAN buses or real steering/braking actuators. All Minimum Risk Maneuver (MRM) logic and electronic power steering (EPS) lockouts are executed strictly in software simulation.

---

## 1. Project Purpose

CABINGUARD-ADI addresses sudden acute medical events behind the wheel (e.g., sudden syncope, tonic-clonic seizures, and cognitive-autonomic deficits) through a **zero-compliance** multimodal sensing and decision-support architecture. "Zero-compliance" signifies that the driver is not required to wear specialized electrodes, wearables, or submit to active manual calibration routines.

The system fuses:
1. **Visual Cabin Cues**: Head pose (pitch/roll/yaw), gaze instability, eye-closure / PERCLOS proxy, and driver motion via non-invasive cameras.
2. **Physiological Signal Processing**: Seated vibration signals (BCG/SCG/EMFi proxies) processed via Normalized Least Mean Squares (NLMS) adaptive filtering for motion artifact removal and cardiorespiratory metric extraction (HR, RR, IBI variability).
3. **Cross-Modal Physical Plausibility & Conflict Checking**: Safeguarding against false positives (e.g., distinguishing slouching vs. syncope collapse, or visual occlusions vs. physiological distress).
4. **Uncertainty-Gated Finite State Machine (FSM)**: Tiered response from Level 1 (Nominal) through Level 2 (Prodromal Warning & 4.0s Cognitive Handshake) to Level 3 (Emergency Simulated MRM Actuation), with automatic **Fail-Silent** fallback under sensor degradation or conflict.

---

## 2. Scientific Integrity Rules (Non-Negotiable)

This codebase enforces strict empirical and reproducible science:

1. **Zero Results Fabrication**:
   - No accuracy, F1-scores, latencies, SNRs, lead times, or throughput numbers are hardcoded or simulated to look complete.
   - Any metric that has not been executed on actual data evaluates to `NOT YET MEASURED` or `NOT AVAILABLE`.
2. **No False Medical Labels on Public Baselines**:
   - Public naturalistic driving datasets (e.g., Ala Mahlak, ADMS, Hugging Face Driver) provide normal driving, distractions (yawning, smoking, phone use), or seated geometry.
   - They are **never** labeled as syncope or seizures.
   - Pathological event scenarios are explicitly categorized and tracked as `SYNTHETIC_FEATURE_LEVEL_SCENARIOS`.
3. **Strict Data Provenance**:
   - Every single sample retains complete lineage tracking: `source_dataset`, `source_sample_id`, `source_subject_id`, `source_session_id`, `is_synthetic`, `scenario_class`, `scenario_phase`, `injected_features`, and `transformation_parameters`.
4. **Guaranteed Data Leakage Prevention**:
   - Partitions (Train / Validation / Test) are split strictly by **Subject ID / Session ID / Video Source** *prior* to any synthetic scenario generation.
   - A synthetic variant derived from a training sample never crosses into the validation or testing sets.

---

## 3. Dataset Setup & Modality Roles

The framework integrates five primary datasets via standardized adapters in `src/cabinguard/datasets/`:

| Dataset Identifier | Canonical Source | Role in CABINGUARD-ADI | Permitted Assumptions & Restrictions |
| :--- | :--- | :--- | :--- |
| **`hf_driver`** | [Hugging Face `bigamma/In_Cabin_Behavior_Driver_safe_drive`](https://huggingface.co/datasets/bigamma/In_Cabin_Behavior_Driver_safe_drive) | In-cabin seated geometry, head/body visual baseline | Gated access; requires user `HF_TOKEN`. No medical labels. |
| **`ala_mahlak`** | [Kaggle `abdoreldeeb/driver-monitoring-system-dataset`](https://www.kaggle.com/datasets/abdoreldeeb/driver-monitoring-system-dataset) | Visual driver monitoring baseline | Inspect directory and label schema. Never infer classes from name alone. |
| **`radar_vitals`** | [Kaggle `cybersimar08/radar-based-vital-sign-monitoring`](https://www.kaggle.com/datasets/cybersimar08/radar-based-vital-sign-monitoring) | Cardiorespiratory signal processing & filter benchmarking | **NOT** a driver incapacitation dataset. Used only to benchmark HR/RR extraction & filtering. |
| **`adl`** | [Kaggle `mocasbu/adl-dataset`](https://www.kaggle.com/datasets/mocasbu/adl-dataset) | Activities of daily living, movement, posture | Role verified dynamically upon schema inspection. |
| **`adms`** | [Kaggle `smartcity12/adms-dataset`](https://www.kaggle.com/datasets/smartcity12/adms-dataset) | Visual confounders, distraction, false-positive benchmarking | Verified classes only (`eye_closure`, `yawning`, `smoking`, `phone_use`, `seatbelt`). Never convert to diagnoses. |

---

## 4. Authentication & Credentials

Credentials must never be committed to source control. Configure your environment using `.env`:

```bash
# Copy template
cp .env.example .env
```

Set the appropriate keys inside `.env`:
```ini
# Hugging Face access token for gated repos:
HF_TOKEN=hf_your_valid_token_here

# Kaggle credentials for automated dataset downloads:
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key

# Custom data and output directories (optional):
CABINGUARD_DATA_ROOT=./data
CABINGUARD_OUTPUT_ROOT=./outputs
```

Alternatively, place your native `kaggle.json` inside `~/.kaggle/kaggle.json` (Linux/macOS) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows).

---

## 5. Environment Installation

### Prerequisites
- Python 3.11+ (Python 3.11 – 3.14 supported)
- NVIDIA GPU with CUDA support recommended (e.g., RTX 3050 Ti up to data-center A10/L4/V100/A5000), but CPU execution is fully supported.

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/cabinguard-ai.git
   cd cabinguard-ai
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate

   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   pip install -e .
   ```

---

## 6. End-to-End Pipeline Execution

The system provides dedicated modular scripts for each phase of the research workflow:

```text
RAW SENSORS / DATASETS
        ↓  (scripts/inspect_datasets.py)
DATASET INVENTORY & AUDIT
        ↓  (scripts/build_manifests.py)
LEAKAGE-SAFE MANIFESTS & SPLITS (70/15/15)
        ↓  (scripts/extract_features.py)
CACHED MULTIMODAL FEATURE STORE (Parquet)
        ↓  (scripts/create_synthetic_scenarios.py)
SYNTHETIC SCENARIO INJECTION (With full provenance)
        ↓  (scripts/train_baselines.py & train_model.py)
MODEL TRAINING (Rule Baseline -> Classical -> TCN -> GRU)
        ↓  (scripts/evaluate_model.py & evaluate_security.py)
BENCHMARKING, LATENCY, SECURITY & SCIENTIFIC AUDIT
```

### Step 1: Inspect Datasets
Inspects available local data, validates licenses, schema, file extensions, and detects potential unsupported claims:
```bash
python scripts/inspect_datasets.py
```
*Outputs: `outputs/reports/dataset_inventory.md` and `dataset_inventory.json`.*

### Step 2: Download Datasets (Optional / Authenticated)
Fetches configured public datasets using Kaggle and Hugging Face APIs:
```bash
python scripts/download_datasets.py
```

### Step 3: Build Leakage-Safe Manifests
Generates subject- and session-aware splits ensuring zero identity leakage across train/validation/test partitions:
```bash
python scripts/build_manifests.py
```
*Outputs: `outputs/reports/split_report.json` and parquet manifests in `data/manifests/`.*

### Step 4: Extract and Cache Features
Processes raw modalities **once** into lightweight tabular/temporal feature arrays (Parquet/NPZ). Video backbones remain frozen:
```bash
python scripts/extract_features.py
```

### Step 5: Generate Synthetic Scenarios
Generates strictly provenance-tracked feature-level medical scenarios (Syncope, Tonic-Clonic Seizure, Cognitive Tremor) alongside distractors (Eye Closure, Yawning) independently per split partition:
```bash
python scripts/create_synthetic_scenarios.py
```

### Step 6: Train Baseline Models
Trains deterministic CABINGUARD baseline (Model A) and fast tabular HistGradientBoosting (Model B):
```bash
python scripts/train_baselines.py
```

### Step 7: Train Deep Temporal Models
Trains lightweight temporal architectures (< 5M parameters) on feature sequences:
```bash
# Train Compact Temporal Convolutional Network (TCN)
python scripts/train_model.py --model tcn

# Train Gated Recurrent Unit (GRU) Comparator
python scripts/train_model.py --model gru
```

### Step 8: Multi-Dimensional Evaluation
Computes comprehensive classification metrics, CABINGUARD safety-specific metrics, lead times, and abstention rates:
```bash
python scripts/evaluate_model.py --checkpoint outputs/checkpoints/best_tcn.pt
```

### Step 9: Security Simulation Harness
Benchmarks the simulated SecOC-style message authentication and freshness verification:
```bash
python scripts/evaluate_security.py
```

### Step 10: Runtime Latency Benchmarking
Measures exact wall-clock latencies (mean, median, p95, p99) for preprocessing, inference, and FSM decision:
```bash
python scripts/benchmark_runtime.py
```

### Complete One-Click Orchestration
To execute the smoke-test pipeline across all phases:
```bash
python scripts/run_full_pipeline.py --smoke-test
```

---

## 7. Mathematical & Engineering Foundations

### 7.1 NLMS Adaptive Filtering for Seated BCG/SCG
Motion artifact cancellation using reference acceleration signals:
$$\hat{s}_{BCG}[n] = x[n] - \sum_{k=0}^{M-1} w_k[n] d[n-k]$$
$$\mathbf{w}[n+1] = \mathbf{w}[n] + \frac{\mu}{\epsilon + \|\mathbf{d}[n]\|_2^2} \hat{s}_{BCG}[n] \mathbf{d}[n]$$
*Parameters: $M = 32$, $f_s = 100\text{ Hz}$, $\mu = 0.05$, $\epsilon = 10^{-6}$.*

### 7.2 Robust Personalized Calibration with Shrinkage
Median and Median Absolute Deviation (MAD) robust estimation regularized toward population baselines:
$$c_{i,s} = \text{median}(x_{i,s,t}), \quad \sigma_{i,s} = 1.4826 \cdot \text{median}(|x_{i,s,t} - c_{i,s}|)$$
$$\tilde{c}_{i,s} = w_i c_{i,s} + (1 - w_i)\mu_{pop,s}, \quad w_i = \frac{N_{good}}{N_{good} + \kappa}$$
*Parameters: $\kappa = 300$, calibration window $= 60\text{ seconds}$.*

### 7.3 Multi-Feature Z-Score Normalization & Severity Bands
$$z_{i,s,t} = O_s \frac{x_{i,s,t} - \tilde{c}_{i,s}}{\tilde{\sigma}_{i,s}}$$
- **NORMAL**: $|z| < 2.0$
- **ELEVATED**: $2.0 \le |z| < 3.0$
- **LARGE**: $3.0 \le |z| < 4.5$
- **EXTREME**: $|z| \ge 4.5$

### 7.4 Synthetic Scenario Envelopes
Temporal evolution modeling without video synthesis:
- **Ramp Envelope**:
  $$g_k(t) = \begin{cases} 0, & t < t_{p,k} \\ \frac{t - t_{p,k}}{t_{a,k} - t_{p,k}}, & t_{p,k} \le t < t_{a,k} \\ 1, & t \ge t_{a,k} \end{cases}$$
- **Target Injections**: $x_s^{(syn)}(t) = (1 - g_k(t))x_s^{(b)}(t) + g_k(t)x_{s,k}^*$
- **Oscillatory Injections** (Seizure: 3–6 Hz, Tremor: 4–8 Hz):
  $$x_s^{(syn)}(t) = x_s^{(b)}(t) + A_s g_k(t) \sin(2\pi f_k t + \phi)$$

### 7.5 Uncertainty and Cross-Modal Conflict Formulation
$$U_t = 1 - \left( w_f C_{fusion,t} + w_a A_{sources,t} (1 - P_{conflict,t}) \right)$$
*Parameters: $w_f = 0.65$, $w_a = 0.35$.*
- **Conflict Trigger**: If camera indicates head pitch $> 45^\circ$ (slump) while seat sensor indicates lumbar contact $> 85\%$, $P_{conflict} = 1.0$, immediately enforcing **Fail-Silent** state.

---

## 8. Hardware Profiles & Small Data-Center Recommendations

The codebase automatically probes system resources at startup via `src/cabinguard/utils/hardware.py` and maps to one of five target profiles:

| Hardware Profile | Target Environment | Training Configuration | Memory Optimizations |
| :--- | :--- | :--- | :--- |
| **`cpu`** | Developer laptops, test VMs | Batch size 16, classical baselines & compact models | No CUDA overhead; cached NumPy / PyArrow arrays. |
| **`single_gpu_8gb`** *(Active Profile)* | Entry GPUs (e.g., RTX 3050 Ti 4GB, RTX 4060 8GB) | Batch size 32, grad accum steps 2, mixed precision (FP16/AMP) | Frozen visual encoders; strictly cached feature stores. |
| **`single_gpu_16gb`** | Workstations (e.g., RTX 4080, V100 16GB, T4) | Batch size 64, grad accum steps 1, TCN / GRU | Full temporal sequence caching in VRAM. |
| **`single_gpu_24gb`** | Server GPUs (RTX 3090/4090, A10, A5000) | Batch size 128, multi-block TCN, larger temporal horizons | High throughput sequence processing. |
| **`multi_gpu`** | Small multi-GPU nodes (2–4 GPUs) | PyTorch DistributedDataParallel (`torchrun`) | Standalone DDP; no bulky orchestration framework required. |

### Running on Multi-GPU Nodes
```bash
torchrun --standalone --nproc_per_node=2 scripts/train_model.py --model tcn --profile multi_gpu
```

---

## 9. Simulated SecOC Security Test Harness

Automotive edge security verification in `src/cabinguard/security/`:
- **Simulated CAN-FD Frames**: Payload authentication with simulated AES-CMAC.
- **Freshness Management**: Synchronized `TripCounter`, `ResetCounter`, and rolling `MsgCounter`.
- **Adversarial Scenarios Tested**:
  1. Valid frames (expected acceptance).
  2. Modified payload with original MAC (tampering detection).
  3. Replayed freshness counter / duplicate message (replay attack rejection).
  4. Out-of-order counter sequence (freshness violation).
  5. Truncated or invalid MAC signatures.

*Note: Labeled `SIMULATED_SECOC_TEST_HARNESS`. Not a formal AUTOSAR certified implementation.*

---

## 10. Automated Unit & Integration Testing

Run the comprehensive test suite verifying schemas, non-leakage splits, numerical math, FSM transitions, and security harnesses:

```bash
pytest tests/ -v
```

### Coverage Highlights:
- `test_dataset_schemas.py`: Schema validation and sample provenance serialization.
- `test_splits.py` & `test_no_leakage.py`: Strict disjoint verification of subjects and sessions across splits.
- `test_filtering.py`: Correctness of Butterworth SOS and NLMS adaptive weight updates.
- `test_scenario_injection.py`: Verification of ramp, target, and oscillatory injection equations.
- `test_uncertainty.py`: Boundary verification of uncertainty and availability formulas.
- `test_fsm.py`: Complete state coverage for Level 1, Level 2, Level 3, and Fail-Silent paths.
- `test_security.py`: Full attack rejection verification in the SecOC simulator.

---

## 11. Known Limitations & Research Boundaries

1. **Synthetic Event Boundary**: Public datasets do not contain actual clinical syncope or epilepsy during driving. While feature transformations obey empirical medical literature bounds (e.g., 3–6 Hz clonic jerks), clinical validation on real patient driving data remains future work.
2. **Proxy Feature Sensitivity**: Head pitch, gaze stability, and PERCLOS proxies derived from monocular in-cabin cameras are sensitive to extreme lighting shifts, severe occlusions, or head wear (sunglasses/caps).
3. **Hardware Actuation Boundary**: Steering lockouts and decelerations are purely simulated logic; physical deployment requires ASIL-D certified hardware and real vehicle bus gateways.

---

## 12. License & Citation

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

```bibtex
@misc{cabinguard2026,
  title={CABINGUARD-ADI: A Zero-Compliance Multimodal In-Cabin Framework for Prodromal Prediction and Fail-Safe Actuation in Acute Driver Incapacitation},
  author={CABINGUARD Research Team},
  year={2026},
  note={TSYP14 Technical Challenge Research Prototype}
}
```

