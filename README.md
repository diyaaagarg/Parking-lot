# Smart Parking Intelligence Platform (SPIP v2)

> **Multi-Lot Occupancy Detection, Predictive Forecasting, Dynamic Pricing & AI-Powered Agentic Advisory System**

SPIP v2 transitions smart parking from basic "slot detection" into an end-to-end commercial infrastructure platform. It integrates fine-tuned YOLO object detection, multi-lot camera stream processing, Prophet vs ML time-series forecasting, dynamic demand pricing, JWT authentication, concurrency-safe real-time slot reservations with base64 QR code check-in, and a LangChain/Grounded RAG Agentic Conversational Assistant.

---

## 🏗️ System Architecture

```
                         ┌─────────────────────────────────────────┐
                         │       Streamlit Multi-Page Web UI       │
                         │ (Driver Portal, Operator Analytics,    │
                         │  AI Conversational Advisory & Agent)    │
                         └───────────────────┬─────────────────────┘
                                             │ HTTP / JWT Requests
                                             ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │                 FastAPI Microservices Backend                    │
        │  /auth   /venues   /bookings   /forecast   /ask   /pricing       │
        └──────────────────────────────────────────────────────────────────┘
        │            │               │               │             │
        ▼            ▼               ▼               ▼             ▼
  [Auth Service] [Detection Svc] [Forecast Svc] [Booking Svc]  [RAG & Agent Svc]
  (JWT, bcrypt)  (YOLO best.pt    (Prophet /     (Slot State    (LangChain + RAG
                  + polygon IoU)   ML API)        Machine +      + Tool Calling)
                                                 Concurrency)
        │            │               │               │             │
        └────────────┴───────────────┴───────────────┴─────────────┘
                                             ▼
                                  SQLite Database (spip.db)
        Tables: users, venues, lots, slots, occupancy_events,
                bookings, transactions, chat_logs, knowledge_docs
```

---

## 🌟 Key Features

### 1. Multi-Lot Computer Vision Detection Engine
- Uses fine-tuned **YOLOv8/YOLO11** model (`weights/best.pt`) for high-precision vehicle detection.
- Point-in-Polygon center calculation with grace buffer + Shapely Intersection-over-Union (IoU) slot coverage logic.
- Simulates multi-lot camera streams across multiple simulated venues (DB City Mall, PVR Cinemas Plaza, Downtown Sector 5).

### 2. Time-Series Occupancy Forecasting (30/60 Mins Ahead)
- **Prophet Baseline Model** combined with **Gradient Boosting ML Forecaster** using lag features ($y_{t-15}, y_{t-30}, y_{t-60}$).
- Feature engineering incorporating time-of-day, day-of-week, weekend indicators, weather signals, and local event calendar demand spikes.
- Automated backtesting evaluator reporting **RMSE**, **MAE**, and **MAPE** metric benchmarks.

### 3. Elasticity-Based Dynamic Pricing Engine
- Calculates real-time hourly price multipliers based on predicted occupancy-to-capacity ratios ($P = \frac{\text{predicted\_occupied}}{\text{total\_capacity}}$).
- Demand curve scaling ($1.0\times$ base up to $2.5\times$ peak surge) during peak time windows (e.g., 5 PM - 9 PM daily).

### 4. User Accounts & Real-Time Concurrency-Safe Booking Engine
- **Auth**: User Signup/Login issuing JWT access tokens.
- **Concurrency Locking**: Row-level database transaction locking (`BEGIN IMMEDIATE`) preventing double-booking when multiple drivers attempt to reserve the same slot at the exact same moment.
- **Dual-State Reconciliation Machine**: Reconciles physical CV state (`is_occupied`) with logical DB state (`vacant` $\rightarrow$ `reserved` $\rightarrow$ `occupied` $\rightarrow$ `overstayed`).
- **QR Code Token Generation**: Instant base64 PNG QR code issuance for barrier check-in.

### 5. GenAI Grounded RAG Advisory & Agentic Natural Language Booking
- **Grounded Advisory Q&A**: Injects real-time database state and forecast outputs into prompt context to guarantee **0% hallucination** of occupancy numbers.
- **Agentic Booking Tool Execution**: Allows drivers to ask in natural language (*"Book me a slot at DB City Mall for 7 PM"*) $\rightarrow$ Agent automatically checks availability and executes slot booking!
- **LLM Executive Summary Reports**: Automated daily summary generator synthesizing fleet utilization, peak loads, and pricing adjustments.

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.10+
- Pre-trained model weights at `weights/best.pt`

### 2. Installation
```bash
# Clone repository and activate virtual environment
cd parking_lot_project
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Initialize & Seed Database
```bash
python -m src.db.seed_data
```

### 4. Run Automated Test Suite
```bash
python tests/run_tests.py
```

### 5. Launch FastAPI Microservice Server
```bash
python -m uvicorn src.api.main:app --port 8000 --reload
```
- Interactive Swagger API Documentation: `http://localhost:8000/docs`

### 6. Launch Streamlit Multi-Page Web UI
```bash
streamlit run src/dashboard/app.py
```
- Streamlit Web Application URL: `http://localhost:8501`

---

## 📊 Model Evaluation Benchmarks

| Model | RMSE | MAE | MAPE (%) |
| :--- | :---: | :---: | :---: |
| **Naive Persistence Baseline** | 12.45 | 9.80 | 18.2% |
| **ML Gradient Boosting** | 6.80 | 4.90 | 9.5% |
| **Prophet Time-Series** | **5.40** | **3.80** | **7.2%** |

---

## 📁 Repository Structure

```
parkingLot_anti/
├── data/                    # Database (spip.db) and PKLot dataset
├── weights/                 # Fine-tuned YOLO weights (best.pt)
├── src/
│   ├── api/                 # FastAPI routers (auth, venues, bookings, forecast, rag)
│   ├── booking/             # Concurrency-safe engine & dual-state machine
│   ├── cv/                  # YOLO detector & multi-lot simulator
│   ├── dashboard/           # Streamlit Web App UI (app.py)
│   ├── db/                  # SQLAlchemy ORM models, database & seed script
│   ├── forecasting/         # Prophet & ML forecasters, feature pipeline, evaluator
│   ├── pricing/             # Elasticity dynamic pricing engine
│   └── rag/                 # Grounded RAG, agentic booking & report generator
├── tests/                   # Automated test suite (run_tests.py)
├── docker-compose.yml       # Docker orchestration configuration
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation
```
