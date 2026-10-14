# 🛡️ DHRITI: Dynamic Healing & Resilience through Intelligent Telemetry & Inference

> An enterprise-grade MLOps & LLMOps platform designed to monitor, optimize, evaluate, and self-heal deployed AI/LLM models in real-time.

---

## 🚀 About the Project
Deploying an AI model is only half the battle; maintaining its performance, controlling latency, and handling model drift is the real challenge[cite: 3]. **DHRITI** acts as an intelligent control and operations layer between user applications and AI models. It optimizes inference costs via semantic caching, monitors output quality, detects performance drift, and automatically triggers self-healing fine-tuning pipelines.

---

## 🏗️ System Architecture & Core Modules

DHRITI is built with an 8-layer modular pipeline design:
1. **Intelligent Gateway (`gateway/main.py`)**: FastAPI-based secure entry point handling request validation and routing.
2. **Semantic Cache (`CACHE_STORE`)**: Avoids redundant LLM generations by recognizing semantically similar prompts.
3. **Quality Scorer (`quality/scorer.py`)**: Automatically scores model responses to gauge accuracy and reliability.
4. **Drift Watchdog (`watchdog/drift.py`)**: Continuously tracks quality degradation and triggers alerts when performance drops.
5. **Self-Healing Pipeline (`healing/pipeline.py`)**: Automatically initiates LoRA/QLoRA fine-tuning protocols upon detecting drift.

---

## 🛠️ Technology Stack
* **Language**: Python 3.10+
* **Framework**: FastAPI (Async API Gateway)
* **Validation**: Pydantic
* **Server**: Uvicorn
* **Version Control**: Git & GitHub

---

## 📁 Project Directory Structure
```text
dhriti-project/
│
├── gateway/
│   ├── __init__.py
│   └── main.py          # FastAPI application & core routing pipeline
│
├── quality/
│   ├── __init__.py
│   └── scorer.py        # AI response quality evaluation logic
│
├── watchdog/
│   ├── __init__.py
│   └── drift.py         # Statistical drift detection & threshold monitoring
│
├── healing/
│   ├── __init__.py
│   └── pipeline.py      # Automated self-healing & fine-tuning trigger
│
├── venv/                # Python Virtual Environment
└── README.md            # Project Documentation