# Intelligent Property Price Prediction & Agentic Real Estate Advisory

**GenAI Capstone — Milestone 2**

A hybrid AI system that predicts Bengaluru property prices using a Machine Learning model and generates structured investment advisory reports using LangGraph + RAG.

---

## Project Overview

This project extends Milestone 1 (ML price prediction) into a full agentic AI system that:

- Predicts property prices using a trained Linear Regression pipeline (R² = 0.83)
- Analyzes market positioning using an LLM (Groq LLaMA 3.3 70B)
- Retrieves relevant market intelligence using RAG (Chroma vectorstore)
- Generates structured investment advisory reports with Buy / Hold / Caution signals

---

## Agent Pipeline

```
Input (Property Details)
    ↓
Market Analysis Node — ML prediction + LLM market analysis
    ↓
Conditional Routing (LangGraph)
    ├── Overpriced  → Retrieval (Chroma) → Advisory Node
    ├── Fair Value  → Advisory Node
    └── Undervalued → Advisory Node
    ↓
Advisory Report (Summary + Comps + Action + Disclaimer)
```

| Price Category | Condition | Signal |
|---|---|---|
| Overpriced | >20% above location avg/sqft | CAUTION |
| Fair Value | within ±20% | HOLD / CONSIDER |
| Undervalued | <20% below location avg/sqft | BUY |

---

## Tech Stack

| Component | Technology |
|---|---|
| ML Model | scikit-learn Linear Regression Pipeline |
| Agent Framework | LangGraph |
| LLM | Groq LLaMA 3.3 70B (free tier) |
| RAG | LangChain + Chroma + FakeEmbeddings |
| UI | Streamlit |
| Dataset | Bengaluru House Data (13,000+ listings) |

---

## Project Structure

```
rea_estate_pred_end_sem/
├── app.py                      # Streamlit UI + LangGraph agent
├── real_estate_ai_agent.py     # Agent pipeline walkthrough (notebook-style)
├── price_model.pkl             # Trained ML model
├── Bengaluru_House_Data.csv    # Dataset
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/harshilv17/rea_estate_pred_end_sem.git
cd rea_estate_pred_end_sem
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

Get a free Groq API key at: https://console.groq.com

```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## Hosted Demo

Deployed on Streamlit Community Cloud — link TBD.

---

## Model Performance (Milestone 1)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 15.96 | 30.00 | 0.83 |
| Random Forest | 14.51 | 30.77 | 0.82 |

Linear Regression was selected for deployment (higher R²).
