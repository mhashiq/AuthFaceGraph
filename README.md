# 🧠 AuthBrain AI Face Analysis Engine & Deep Learning Research Platform
### *Enterprise Product Details & Technical Reference Report*

> **Confidential Technical Specifications & Architectural Documentation**  
> **Author:** Senior AI Architect & Computer Vision Scientist  
> **Version:** 2.0.0 (Phase 1 Integration Complete)  
> **Target Audience:** Core Engineering Teams, ML Researchers, Security Auditors  

---

## 📋 Executive Summary
AuthBrain is an enterprise-grade, privacy-first **facial behavior analysis and deep learning research platform**. It processes live webcam streams to perform high-frequency attention monitoring, drowsiness detection, and facial expression analysis. 

The system operates via two concurrent analytical engines:
1. **An Expert Rule Engine:** Providing instant, deterministic, and auditable risk scoring based on physiological aspect ratios (EAR, MAR) and head pose angles.
2. **A Modular Deep Learning Plugin Pipeline:** Running state-of-the-art Convolutional Neural Networks (CNNs) for facial expression classification, Geometric Action Unit (FACS) estimators, and Graph Neural Networks (GNNs) mapping MediaPipe FaceMesh coordinates to spatial graph attention structures.

---

## 🛰 High-Level System Architecture

```
                                 ┌──────────────────────────────────────────────┐
                                 │              Client (React SPA)              │
                                 │  Webcam Stream → Consented Frame Capture     │
                                 │  Live HUD Canvas ← Cybernetic GNN Visualizer │
                                 └──────────────────────┬───────────────────────┘
                                                        │ WSS Protocol (Binary JPEG ↑ / JSON Result ↓)
                                                        ▼
                                 ┌──────────────────────────────────────────────┐
                                 │         FastAPI Web Handler (Uvicorn)        │
                                 │  Middlewares: CORS, JWT Verification, JWT-WSS │
                                 └──────────────────────┬───────────────────────┘
                                                        │ ThreadPoolExecutor Worker
                                                        ▼
                                 ┌──────────────────────────────────────────────┐
                                 │        FaceAnalysisPipeline (CV Loop)        │
                                 │  1. Image Decode & Normalization (OpenCV)    │
                                 │  2. Landmark Coordinates extraction (478 pts)│
                                 │  3. Gaze, EAR, MAR, & Head Pose Estimators   │
                                 └──────────────────────┬───────────────────────┘
                                                        │
                              ┌─────────────────────────┴─────────────────────────┐
                              ▼                                                   ▼
                ┌───────────────────────────┐                       ┌───────────────────────────┐
                │    Expert Rule Engine     │                       │   Deep Learning Engine    │
                │                           │                       │                           │
                │  - 10 Auditable Rules     │                       │  - Model Registry Hub     │
                │  - Composite Scorer       │                       │  - HSEmotion (ONNX model)  │
                │  - Feature Attributions   │                       │  - GNN GAT (478-node)     │
                │  - Risk Severity Sorter   │                       │  - FACS Action Units      │
                │                           │                       │  - GNNExplainer (XAI)     │
                └─────────────┬─────────────┘                       └─────────────┬─────────────┘
                              │                                                   │
                              └─────────────────────────┬─────────────────────────┘
                                                        ▼
                                 ┌──────────────────────────────────────────────┐
                                 │          Database Storage & ORM              │
                                 │ SQLite / PostgreSQL — SQLAlchemy Async ORM   │
                                 └──────────────────────────────────────────────┘
```

---

## 🛠 Backend Deep Learning Architecture

### 1. Model Registry Hub & Plugin Infrastructure
- **File:** [registry.py](file:///Users/mdmehedihassan/Desktop/Projects/AuthBrain_AI_Face_Analysis/backend/app/dl/registry.py)
- Features a **thread-safe, singleton-based model registry** that permits dynamic model loading, caching, and health auditing.
- Provides abstract base classes (`EmotionModelBase`, `GNNModelBase`) in [base.py](file:///Users/mdmehedihassan/Desktop/Projects/AuthBrain_AI_Face_Analysis/backend/app/dl/base.py) ensuring consistent interfaces for inputs, outputs, and latencies.

### 2. Pretrained Emotion Recognition Pipeline
- **HSEmotion ONNX Recognizer:**
  - **File:** [hsemotion.py](file:///Users/mdmehedihassan/Desktop/Projects/AuthBrain_AI_Face_Analysis/backend/app/dl/emotion/hsemotion.py)
  - Loads an `enet_b0_8_best_afew` backbone optimized with **ONNX Runtime** for low latency CPU execution (typically 8–15 ms).
  - Translates prediction arrays directly to our normalized emotion categories (`neutral`, `happy`, `sad`, `surprise`, `fear`, `disgust`, `anger`, `contempt`) using a robust label translation mapping.
- **EfficientFace Pipeline:**
  - **File:** [efficientface.py](file:///Users/mdmehedihassan/Desktop/Projects/AuthBrain_AI_Face_Analysis/backend/app/dl/emotion/efficientface.py)
  - Features a PyTorch-based implementation of EfficientFace, pulling cached parameters from HuggingFace (`Antigravity/efficientface-affectnet8`).
- **Ensemble Combination Layer:**
  - **File:** [ensemble.py](file:///Users/mdmehedihassan/Desktop/Projects/AuthBrain_AI_Face_Analysis/backend/app/dl/emotion/ensemble.py)
  - Merges confidence probabilities from all active classifiers using a temperature-scaled weighted average consensus.
  - Computes a normalized Shannon entropy **Disagreement Score** and **Uncertainty Factor** to flag multi-model predictions that lack statistical consensus.

### 3. Landmark Graph Constructor & GNN Classifier
- **Face Graph Constructor:**
  - **File:** [constructor.py](file:///Users/mdmehedihassan/Desktop/Projects/AuthBrain_AI_Face_Analysis/backend/app/dl/graph/constructor.py)
  - Translates 478 MediaPipe 3D coordinates into a topological graph.
  - Each node feature vector $x_i \in \mathbb{R}^{10}$ represents:
    - 3D Coordinates $(x, y, z)$
    - Displacement vectors from baseline calibration
    - Rolling velocity vector $(\Delta x, \Delta y, \Delta z)$ over the temporal window
    - One-hot structural region categorization (left eye, right eye, forehead, lips, nose, cheeks, jaw)
  - Automatically builds edges using a $k$-Nearest Neighbors ($k$-NN) spatial topology ($k=6$).
- **Graph Attention Network (GAT):**
  - **File:** [gat.py](file:///Users/mdmehedihassan/Desktop/Projects/AuthBrain_AI_Face_Analysis/backend/app/dl/graph/gat.py)
  - Runs a 3-layer PyTorch Geometric GAT network utilizing multi-head attention weights ($heads=4$) to capture geometric deformations.

### 4. FACS Action Units & Explainable AI (XAI)
- **FACS Action Unit Estimator:**
  - **File:** [au_estimator.py](file:///Users/mdmehedihassan/Desktop/Projects/AuthBrain_AI_Face_Analysis/backend/app/dl/action_units/au_estimator.py)
  - Evaluates geometric deformation criteria (distances, normalized angles) across key muscle zones to infer standard FACS Action Units:
    - **AU1** (Inner Brow Raiser), **AU2** (Outer Brow Raiser), **AU4** (Brow Lowerer)
    - **AU12** (Lip Corner Puller - Smile), **AU15** (Lip Corner Depressor - Frown)
    - **AU25** (Lips Part), **AU26** (Jaw Drop), **AU45** (Blink / Closure)
- **GNN Explainer Layer:**
  - **File:** [gnn_explainer.py](file:///Users/mdmehedihassan/Desktop/Projects/AuthBrain_AI_Face_Analysis/backend/app/xai/gnn_explainer.py)
  - Performs post-hoc node attribution profiling by extracting node embeddings from GAT attention blocks. Returns normalized continuous attribution weights representing node importance.

---

## 💻 Web & Visualization Layer

### 1. Cybernetic Holographic GNN Visualizer
- **File:** [MetricsPanel.tsx](file:///Users/mdmehedihassan/Desktop/Projects/AuthBrain_AI_Face_Analysis/frontend/src/components/dashboard/MetricsPanel.tsx)
- Renders an interactive, futuristic cybernetic face mesh visualization using an optimized HTML5 Canvas loop.
- **Visual Specifications:**
  - **Robotic Outline:** Wireframe connections mapped in clean, high-contrast digital white lines (`rgba(255, 255, 255, 0.18)`), which dynamically transition into neon cyan-400 (`rgba(34, 211, 238, ...)`) when edge attention weights spike.
  - **Robotics-style Nodes:** Dormant landmarks drawn as crisp white digital points.
  - **Neon Target Reticles (XAI Hotspots):** Active node landmarks (GNN attributions) scale up dynamically in glowing rose-500. Nodes with high activations (`importance > 0.4`) draw concentric target reticles and crosshairs lines, emphasizing GNN attention highlights.

### 2. Multi-Model Selector & Real-Time Plots
- Built a modular React layout:
  - **Ensemble Panel:** Model registry manager with model selector trigger, letting users inspect specific latencies, disagreement entropy, and prediction probability bar charts.
  - **Radar Chart & Timeline:** Live Recharts radar showing the 8-class probability distribution, and a rolling timeline showing continuous emotional trends over time.
  - **Action Unit Intensity Meters:** Real-time intensity indicator bars for the 8 FACS Action Units.

---

## 🗄 Database Schema & persistence

The database system supports PostgreSQL (Production) and SQLite (Local Development) via an asynchronous SQLAlchemy ORM layer.

```sql
-- PostgreSQL Session Aggregates Schema
CREATE TABLE analysis_sessions (
    id UUID PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID REFERENCES users(id),
    org_id UUID REFERENCES organizations(id),
    status VARCHAR(50) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    total_frames INTEGER DEFAULT 0,
    total_blinks INTEGER DEFAULT 0,
    avg_ear FLOAT DEFAULT 0.0,
    avg_head_yaw FLOAT DEFAULT 0.0,
    avg_head_pitch FLOAT DEFAULT 0.0,
    avg_fatigue_score FLOAT DEFAULT 0.0,
    avg_focus_score FLOAT DEFAULT 0.0,
    max_fatigue_score FLOAT DEFAULT 0.0,
    avg_stress_risk FLOAT DEFAULT 0.0,
    face_quality_score FLOAT DEFAULT 0.0,
    dominant_attention_state VARCHAR(100),
    avg_inference_time_ms FLOAT DEFAULT 0.0
);
```

---

## 📂 Directory Structure

```
AuthBrain_AI_Face_Analysis/
├── start.sh                        # Unified backend + frontend execution script
├── docker-compose.yml              # Multi-container orchestrator (PostgreSQL, FastAPI, Vite)
├── .env / .env.example             # Configuration variables
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI application and lifespan handlers
│   │   ├── analysis/
│   │   │   ├── pipeline.py         # Frame processing orchestrator
│   │   │   ├── face_detector.py    # MediaPipe FaceLandmarker task runner
│   │   │   ├── eye_analyzer.py     # Aspect ratios, blink counts, gaze directions
│   │   │   ├── head_pose.py        # SolvePnP pose solvers
│   │   │   ├── mouth_analyzer.py   # Mouth Aspect Ratio, smile intensities, yawns
│   │   │   ├── behavior_tracker.py # Movement thresholds, temporal smoothing
│   │   │   └── quality_scorer.py   # Contrast, sharpness, illumination models
│   │   ├── dl/
│   │   │   ├── base.py             # Abstract base model wrappers
│   │   │   ├── registry.py         # Model registration singleton hub
│   │   │   ├── engine.py           # Deep Learning aggregate processor
│   │   │   ├── emotion/
│   │   │   │   ├── hsemotion.py    # HSEmotion ONNX wrapper
│   │   │   │   ├── efficientface.py# EfficientFace HuggingFace downloader
│   │   │   │   └── ensemble.py     # Probability consensus layer
│   │   │   ├── graph/
│   │   │   │   ├── constructor.py  # FaceMesh-to-Graph compiler (KNN)
│   │   │   │   └── gat.py          # PyTorch Geometric GAT & GCN classifiers
│   │   │   ├── action_units/
│   │   │   │   └── au_estimator.py # Geometric FACS estimator
│   │   │   └── xai/
│   │   │       └── gnn_explainer.py# Attention weight extractor
│   │   ├── expert_system/
│   │   │   ├── rules.py            # 10 deterministic expert rules
│   │   │   ├── scorer.py           # Fatigue & focus aggregation math
│   │   │   └── explainer.py        # Rule-based XAI attributions
│   │   ├── api/
│   │   │   ├── routes/             # REST endpoints (auth, sessions, models)
│   │   │   └── websocket/          # WebSocket frame parser
│   │   ├── core/
│   │   │   ├── config.py           # Pydantic Settings configuration parser
│   │   │   ├── database.py         # Async DB connectors
│   │   │   └── logging.py          # Structlog configuration mapping
│   │   └── models/
│   │       ├── schemas.py          # JSON Pydantic data schemas
│   │       └── db_models.py        # SQLAlchemy relational schemas
│   ├── tests/
│   │   ├── unit/                   # 30 passing PyTest unit tests
│   │   └── conftest.py             # PyTest configurations & mocks
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── ConsentPage.tsx     # Explicit user consent wall
│   │   │   └── Dashboard.tsx       # Live visualization dashboard
│   │   ├── components/
│   │   │   ├── webcam/
│   │   │   │   └── CameraFeed.tsx  # Frame grabber & binary socket loop
│   │   │   └── dashboard/
│   │   │       ├── MetricsPanel.tsx# Cyber GNN visualizer & metric cards
│   │   │       ├── EnsemblePanel.tsx# Model selectors & latencies
│   │   │       ├── EmotionRadarChart.tsx # Recharts emotional radar
│   │   │       └── ActionUnitsPanel.tsx  # FACS intensity meters
│   │   ├── store/
│   │   │   └── analysisStore.ts    # Zustand global state container
│   │   └── types/
│   │       └── analysis.ts         # TypeScript bindings matching backend schemas
│   └── package.json
└── models/
    └── face_landmarker.task        # MediaPipe Task file (Downloaded on first boot)
```

---

## 🏃 Run & Installation Specifications

### 1. One-Command Quick Start
Free all required ports, load environment variables, spin up FastAPI, and run the Vite client server:
```bash
./start.sh
```

### 2. Environment Variables (`.env` configuration)
Create a `.env` in the root workspace folder:
```ini
# Deep Learning Configuration
DL_ENABLED=true
DL_DEVICE=cpu
DL_EMOTION_MODELS=["hsemotion","efficientface"]
DL_GNN_ENABLED=true
DL_XAI_ENABLED=true
DL_GRAPH_EDGE_STRATEGY=knn
DL_GRAPH_KNN_K=6
```

---

*AuthBrain represents a clean, production-grade fusion of classical physiological signal processing and advanced geometric deep learning. Built for explainable, responsible, privacy-first computer vision.*
