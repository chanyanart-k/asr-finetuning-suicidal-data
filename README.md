
# ASR Fine-Tuning for Suicidal-Related Term Detection in Thai

> **Enhancing Automatic Speech Recognition (ASR) for mental health crisis detection in Thai language** — a domain-adapted fine-tuning pipeline built on top of state-of-the-art Thai ASR models.

---

## Table of Contents

- [Background](#background)
- [Objective](#objective)
- [System Architecture](#system-architecture)
- [Models](#models)
- [Dataset](#dataset)
- [Suicidal Term Taxonomy](#suicidal-term-taxonomy)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Demo App](#demo-app)
- [Ethical Considerations](#ethical-considerations)
- [Acknowledgements](#acknowledgements)

---

## Background

In Thailand, mental health remains a critical public health concern. According to the 2025 Thai Health Report, **13.4 million Thais** have experienced a mental health condition at least once in their lifetime, and suicide deaths reached **5,172 cases** (7.94 per 100,000 population) in 2023 — an average of 14 deaths per day.

The **Voicebot** (accessible via the 1323 mental health hotline) is an AI-powered triage system that collects patient responses to a 9-question mental health assessment and prioritizes callers by severity. This system relies on an ASR model to transcribe user speech into text before passing it to a downstream classification model.

However, general-purpose ASR models frequently **misrecognize suicidal terminology**, leading to erroneous transcriptions of high-risk keywords and reducing the downstream classifier's ability to accurately flag at-risk individuals. In 2022, approximately **400,000 calls per year** were received by the hotline, of which only ~100,000 were answered (25%), with 3.34% classified as severe cases.

---

## Objective

> **To replace the current ASR model with a domain-fine-tuned alternative that enhances detection of suicidal terminology in Thai telephone-quality speech.**

Fine-tuning addresses four key value areas:

| Value Area | Problem | Solution | Impact |
|---|---|---|---|
| **High-Risk Keyword Recall** | ASR drops suicidal terms | Domain fine-tuning on suicidal vocabulary | Reduces False Negatives in critical cases |
| **Speech Variability** | Crying, whispering, phone noise degrade accuracy | Trained on real-world noisy clinical audio | Captures unclear speech more robustly |
| **Downstream Model Quality** | ASR errors propagate to the classifier | More accurate transcripts as input | Improves end-to-end system accuracy |
| **Operational Efficiency** | Severe cases not prioritized due to transcription errors | Severe callers receive correct urgency signals | Reduces drop-off from high-risk group |

---

## System Architecture

```
  Audio (Private dataset)
        │
        ▼
┌─────────────────────┐
│   ASR Model (Ours)  │  ← Fine-tuned on Thai suicidal speech domain
│  (Speech → Text)   │
└─────────────────────┘
        │
        ▼  "อยากตายค่ะ"  (Thai: "I want to die")
┌─────────────────────┐
│ Classification Model│  ← Severity scoring
└─────────────────────┘
        │
        ▼
  Priority Queue → Routed to mental health professional
```

**Training Workflow:**

```
Data Preparation → Model Fine-Tuning → Evaluation on Test Set → Deployment / Demo
```

---

## Models

Two base model families were fine-tuned on the domain-specific dataset:

### 1. `biodatlab/distill-whisper-large-v3` (Whisper-based)

Based on OpenAI's **Whisper** Transformer (Seq2Seq) architecture, further distilled via the Thonburian Whisper paper using Knowledge Distillation (Teacher-Student training with KL Divergence loss).

| Property | Description |
|---|---|
| Architecture | Transformer Seq2Seq (Encoder-Decoder) |
| Model Size | 3.24 GB |
| Parameters | 809M (0.8B) |
| Base | `openai/whisper-large-v3` → `biodatlab/distill-whisper-large-v3` |
| Pre-trained on | 1,500 hours of Thai audio |

**Fine-tuning hyperparameters (Whisper):**

| Parameter | Value |
|---|---|
| `num_process_gpu` | 4 |
| `train_batch_size` | 4 |
| `gradient_accumulation_step` | 4 (global batch = 64) |
| `precision` | fp16 |
| `learning_rate` | 1e-5 |


---

### 2. `scb10x/typhoon-asr-realtime` (FastConformer-Transducer)

Based on NVIDIA's **FastConformer-Transducer (RNN-T)** architecture — a streaming-compatible model designed for low-latency real-time inference.

| Property | Description |
|---|---|
| Architecture | FastConformer-Transducer (RNNT) |
| Model Size | 462 MB |
| Parameters | 115M |
| Base | `nvidia/stt_en_fastconformer_transducer_large` → `scb10x/typhoon-asr-realtime` |
| Pre-trained on | 11,000 hours of general Thai data |

**Fine-tuning strategy (Typhoon — 2-Stage Curriculum):**

| Stage | Description | LR | Epochs | Modules |
|---|---|---|---|---|
| **Stage 1** | Global Adaptation (acoustic shift) | 1e-5 | 10 | All |
| **Stage 2** | Linguistic Adaptation (grammar specialization) | 1e-3 | 15 | Decoder + Joint |

---

## Dataset

Audio data was sourced from the **Private dataset** of 5,760 de-identified users, stratified by suicide risk level:

| Risk Level | Users | Files | 
|---|---|---|
| High Risk | 4,032 (70%) | 31,761 | 
| Medium Risk | 1,152 (20%) | 7,987 | 
| Low Risk | 576 (10%) | 2,552 |
| **Total** | **5,760** | **43,200** | 

**Preprocessing pipeline:**
1. Raw audio files concatenated per user
2. **Voice Activity Detection (VAD)** using `pyannote/speaker-diarization-3.1` to remove silence
3. Chunked into segments ≤ 10 seconds each
4. 43,200 files sampled (~120 hours) for ground-truth human annotation.


**Train / Validation / Test split:**

| Split | Files | Hours | % |
|---|---|---|---|
| Train | 41,472 | 100 | 85% |
| Validation | 1,517 | 4 | 3% |
| Test | 5,000 | 14 | 12% |
| **Total** | **47,944** | **118** | 100% |

> Sensitive data is restricted by Access Control List (ACL) policy and is not publicly released.

---

## Suicidal Term Taxonomy

A domain lexicon of **60 suicidal terms** in Thai was compiled and grouped into 4 clinical categories:

| Category | Thai Terms (Count) | Description |
|---|---|---|
| **General Psychological Distress (GPD)** | 6 terms | Reflects negative self-concept and hopelessness (e.g., ฆ่าตัวตาย, ไม่มีค่า, สิ้นหวัง) |
| **Passive Death Wish (PDW)** | 6 terms | Desire for non-existence without explicit intent (e.g., หายไป, หลับไปเลย, อุบัติเหตุ) |
| **Self-harm (SH)** | 13 terms | Self-injury or bodily harm (e.g., กรีด, ทำร้ายตัวเอง, เลือดออก, มีด) |
| **Suicidal Planning/Attempt (SPA)** | 32 terms | Explicit suicidal ideation, methods, or preparatory behaviors (e.g., กระโดด, วางแผน, แขวนคอ, ปืน) |

These terms serve as the target vocabulary for recall-based evaluation.

---

## 📊 Evaluation Metrics

Three complementary evaluation dimensions were used:

### Error Rate
- **Word Error Rate (WER)** ↓ — measures transcription errors via Substitution, Deletion, Insertion
- **Character Error Rate (CER)** ↓

### Term-Level Recall (Medical Term Recall — MTR)
- **Micro Recall (Sensitivity)** ↑ — overall recall weighted by term frequency; primary metric
- **Macro Recall** — average recall across all terms (each term contributes equally)
- **Rare Macro Recall (≤K)** — recall on infrequent terms in the test set

### Utterance-Level Recall
- **Recall** ↑ — proportion of at-risk utterances correctly flagged
- Minimize **False Negatives (FN)** — missing a high-risk statement is the most critical failure

---

## Results

Evaluated on the **5,000-file test set (14 hours):**

| Model | WER ↓ | CER ↓ | MTR / Micro Recall ↑ | Macro Recall ↑ | Rare Recall (≤10) ↑ |
|---|---|---|---|---|---|
| `biodatlab/distill-whisper-large-v3` (base) | 57.725 | 39.799 | 0.389 | 0.363 | 0.24 |
| **+ Domain Fine-tuning (Ours — Whisper)** | **25.541** | **18.158** | **0.903** | **0.787** | **0.67** |
| `scb10x/typhoon-asr-realtime` (base) | 50.681 | 37.285 | 0.382 | 0.394 | 0.39 |
| **+ Domain Fine-tuning (Ours — Typhoon)** | **36.215** | **27.821** | **0.733** | **0.557** | **0.42** |

**Key findings:**
- The fine-tuned **Whisper model** achieves a **132% relative improvement** in Micro Recall (0.389 → 0.903) and reduces WER by more than half
- The fine-tuned **Typhoon model** improves Micro Recall by 92% (0.382 → 0.733) while remaining real-time capable
- Both models show significant improvements on rare suicidal terms, which are the most clinically critical

---

## Repository Structure

```
asr-finetuning-suicidal-data/
├── app.py                  # Streamlit demo application (upload, record, benchmark)
├── main.py                 # Entry point / fine-tuning orchestration
├── processing_utils.py     # Audio preprocessing utilities (VAD, chunking, resampling)
├── run_typhoon_asr.py      # Typhoon ASR model loader and inference runner
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project configuration
├── uv.lock                 # Dependency lock file (uv)
├── .python-version         # Python version pin
└── .gitignore
```

---

## Installation

**Prerequisites:** Python 3.10+, CUDA-compatible GPU (for training)

```bash
# Clone the repository
git clone https://github.com/chanyanart-k/asr-finetuning-suicidal-data.git
cd asr-finetuning-suicidal-data

# Install dependencies using pip
pip install -r requirements.txt

# Or using uv (recommended)
uv sync
```

**Key dependencies:**

| Package | Purpose |
|---|---|
| `nemo_toolkit[asr]==2.0.0` | NeMo ASR framework (Typhoon model training & inference) |
| `streamlit>=1.29` | Demo web application |
| `librosa>=0.11.0` | Audio loading and processing |
| `pythainlp==5.1.2` | Thai NLP utilities |
| `pydub==0.25.1` | Audio format conversion |
| `soundfile==0.12.1` | Audio I/O |

---

## Usage

### Running the Demo App

```bash
streamlit run app.py
```

Configure your model paths in `.streamlit/secrets.toml`:

```toml
[gdrive]
FINETUNED_FILE_ID = "<your_google_drive_file_id>"
BASE_FILE_ID = "<your_google_drive_file_id>"
FINETUNED_MODEL_PATH = "path/to/finetuned_model.nemo"
BASE_MODEL_PATH = "path/to/base_model.nemo"
```


### Benchmarking Models

Use the **Benchmark** tab in the Streamlit app to compare transcription output side-by-side between the **base `scb10x/typhoon-asr-realtime`** and the **fine-tuned Typhoon model** on any uploaded audio file.
 
The fine-tuned Whisper model is not included in the demo app due to its size. It is intended for offline/batch evaluation only.

---

## Demo App

The Streamlit application provides three modes:

| Tab | Description |
|---|---|
| **⬆️ Upload audio file** | Upload a WAV/MP3/M4A/FLAC/OGG/AAC file and transcribe with the fine-tuned model |
| **🎤 Record from mic** | Record directly from the browser microphone and transcribe in real-time |
| **📊 Benchmark** | Run the same audio through both the base model and fine-tuned model side-by-side for comparison |

**Note:** The demo uses the **fine-tuned `scb10x/typhoon-asr-realtime` model only**. The fine-tuned Whisper model (`biodatlab/distill-whisper-large-v3`) is excluded from the demo due to its large size (3.24 GB / 809M parameters), which makes it impractical for lightweight deployment. For full benchmark results including the Whisper model, refer to the [Results](#results) section.

---

## Ethical Considerations

This project works with highly sensitive clinical data related to suicide and mental health. The following safeguards are in place:

- All audio data is **de-identified** prior to use — no personally identifiable information (PII) is retained
- Data access is governed by an **Access Control List (ACL) policy**; raw audio data is **not publicly released**
- The model is intended to **augment** — not replace — trained mental health professionals
- False negatives (missed high-risk utterances) are treated as the primary failure mode to minimize


> This tool is not a substitute for professional psychiatric evaluation. If you or someone you know is in crisis, please contact the **1323 mental health hotline** (Thailand) or your local emergency services.

---

## Acknowledgements

This work was conducted under the **Digital Health Center (DHC)** in collaboration with **AIMET**.

The fine-tuned models are built on top of:
- **Thonburian Whisper** — Robust Fine-tuned and Distilled Whisper for Thai ([paper](https://aclanthology.org/2024.icnlsp-1.17.pdf))
- **Typhoon-ASR-Realtime** — SCB10X Thai ASR model ([paper](https://arxiv.org/abs/2601.13044))
- **Distil-Whisper** — Robust Knowledge Distillation via Large-Scale Pseudo Labelling ([paper](https://arxiv.org/abs/2311.00430))
- **FastConformer** — Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition ([paper](https://arxiv.org/abs/2305.05084))
- **pyannote/speaker-diarization-3.1** — Voice Activity Detection ([Model]https://huggingface.co/pyannote/speaker-diarization-3.1)

Compute resources provided by the **LANTA HPC** supercomputing cluster.

---

**Author:** Chanyanart K.
