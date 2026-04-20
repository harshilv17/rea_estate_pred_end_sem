# -*- coding: utf-8 -*-
"""
# Intelligent Property Price Prediction & Agentic Real Estate Advisory

GenAI Capstone — Milestone 2 | Agentic AI Real Estate Advisory System

### 🔧 Pipeline Overview:

Input → ML Model → Market Analysis (LLM) → Conditional Routing →
    → Overpriced  → Retrieval (Chroma) → Advisory (LLM) → Caution Report
    → Fair Value  → Advisory (LLM) → Balanced Report
    → Undervalued → Advisory (LLM) → Opportunity Report

Step 1: Price Prediction
- Property data fed to trained sklearn Linear Regression pipeline
- Outputs estimated price in Lakhs (INR)

Step 2: Market Analysis (LLM)
- LLM interprets the price vs location average
- Key signals: price_per_sqft, location_avg_ppsf, bhk, total_sqft

Step 3: Conditional Routing (LangGraph)
- Overpriced  (>20% above location avg/sqft) → RAG Retrieval → Advisory
- Fair Value  (within ±20%)                  → Direct Advisory
- Undervalued (<20% below location avg/sqft) → Direct Advisory

Step 4: Advisory Generation
- Structured report: Summary + Comps + Investment Action + Disclaimer
"""

# ── 1. LLM Setup (Groq API) ─────────────────────────────────────────────────

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile"
)

response = llm.invoke("In one sentence, what makes Bengaluru real estate unique?")
print(response.content)


# ── 2. Load Trained ML Model ────────────────────────────────────────────────

import pickle

with open("price_model.pkl", "rb") as f:
    model = pickle.load(f)

print("ML model loaded:", type(model))


# ── 3. Load Dataset & Compute Location Benchmarks ──────────────────────────

import pandas as pd
import numpy as np

data = pd.read_csv("Bengaluru_House_Data.csv")
data = data.drop(["society", "availability"], axis=1, errors="ignore")
data = data.dropna()


def convert_sqft(x):
    try:
        if "-" in str(x):
            a, b = x.split("-")
            return (float(a) + float(b)) / 2
        return float(x)
    except:
        return None


data["total_sqft"] = data["total_sqft"].apply(convert_sqft)
data = data.dropna()

extracted = data["size"].str.split().str[0]
data["bhk"] = pd.to_numeric(extracted, errors="coerce")
data.dropna(subset=["bhk"], inplace=True)
data["bhk"] = data["bhk"].astype(int)
data = data.drop("size", axis=1)

data["price_per_sqft"] = (data["price"] * 100000) / data["total_sqft"]

# Per-location average price/sqft — used for market positioning
location_avg_ppsf = data.groupby("location")["price_per_sqft"].mean().to_dict()
global_avg_ppsf = float(data["price_per_sqft"].mean())

print(f"Loaded {len(data):,} listings | {data['location'].nunique()} unique locations")


# ── 4. Prediction & Market Helper Functions ─────────────────────────────────


def predict_price(property_data: dict) -> float:
    input_df = pd.DataFrame([property_data])
    return float(model.predict(input_df)[0])


def get_price_category(location: str, total_sqft: float, predicted_price: float) -> str:
    ppsf = (predicted_price * 100000) / total_sqft
    avg_ppsf = location_avg_ppsf.get(location, global_avg_ppsf)
    ratio = ppsf / avg_ppsf
    if ratio > 1.20:
        return "Overpriced"
    elif ratio < 0.80:
        return "Undervalued"
    return "Fair Value"


def get_comparable_properties(location: str, bhk: int) -> dict:
    comps = data[(data["location"] == location) & (data["bhk"] == bhk)]["price"]
    if len(comps) < 3:
        comps = data[data["bhk"] == bhk]["price"]
    return {
        "min": round(float(comps.min()), 2),
        "max": round(float(comps.max()), 2),
        "avg": round(float(comps.mean()), 2),
        "count": int(len(comps)),
    }


sample = {
    "area_type": "Super built-up  Area",
    "location": "Whitefield",
    "total_sqft": 1500.0,
    "bath": 2.0,
    "balcony": 1.0,
    "bhk": 3,
}

print("Sample prediction: ₹", round(predict_price(sample), 2), "Lakhs")
print(
    "Price category:",
    get_price_category(sample["location"], sample["total_sqft"], predict_price(sample)),
)


# ── 5. Agent State Definition ───────────────────────────────────────────────

from typing import TypedDict, List, Dict


class AgentState(TypedDict):
    property_data: Dict
    predicted_price: float
    price_category: str
    market_analysis: str
    retrieved_docs: List[str]
    advisory_report: str


# ── 6. Market Analysis Node (LLM) ──────────────────────────────────────────


def market_analysis_node(state: AgentState) -> dict:
    prop = state["property_data"]
    predicted_price = predict_price(prop)
    price_category = get_price_category(
        prop["location"], prop["total_sqft"], predicted_price
    )

    ppsf = (predicted_price * 100000) / prop["total_sqft"]
    avg_ppsf = location_avg_ppsf.get(prop["location"], global_avg_ppsf)

    prompt = f"""
You are a senior real estate market analyst for Bengaluru, India.

Property Details:
- Location: {prop['location']} | Area Type: {prop.get('area_type')}
- Size: {prop['total_sqft']} sqft | BHK: {prop['bhk']} | Bathrooms: {prop.get('bath')}

ML Predicted Price: ₹{predicted_price:.2f} Lakhs
Price per Sqft: ₹{ppsf:,.0f} vs Location Avg: ₹{avg_ppsf:,.0f}
Price Category: {price_category}

Instructions:
- Focus on location and price-per-sqft signal
- Do NOT add assumptions beyond the given data
- Max 80 words

Output EXACTLY in this format:
Market Position:
Key Value Drivers:
"""

    response = llm.invoke(prompt)
    return {
        "predicted_price": predicted_price,
        "price_category": price_category,
        "market_analysis": response.content,
    }


state = {
    "property_data": sample,
    "predicted_price": 0.0,
    "price_category": "",
    "market_analysis": "",
    "retrieved_docs": [],
    "advisory_report": "",
}

out = market_analysis_node(state)
print(out["market_analysis"])


# ── 7. RAG Knowledge Base (Chroma) ─────────────────────────────────────────

documents = [
    "Electronic City and Whitefield are major IT hubs with high demand and above-average price appreciation.",
    "Properties near Bengaluru metro stations command a 10-20% premium and show stronger long-term appreciation.",
    "Sarjapur Road and Hebbal have seen 15-20% appreciation in 3 years due to IT corridor expansion.",
    "Super built-up area includes 20-30% common area; always evaluate carpet area for true value assessment.",
    "Prime Bengaluru areas (Indiranagar, Koramangala) average ₹8,000-15,000/sqft; developing areas ₹4,000-7,000/sqft.",
    "RERA compliance and Occupancy Certificate (OC) are mandatory for safe property transactions in Karnataka.",
    "Ready-to-move properties command 5-15% premium over under-construction due to immediate occupancy and no GST.",
    "Properties below ₹50L in peripheral Bengaluru show higher rental yield (3-4%) vs premium properties (1.5-2.5%).",
    "Overpriced properties in saturated Bengaluru micro-markets take 6-12 months longer to resell.",
    "A property priced more than 20% above the location average per-sqft rate is a high negotiation risk.",
]

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings


def setup_rag() -> Chroma:
    return Chroma.from_texts(documents, FakeEmbeddings(size=384))


vectorstore = setup_rag()


def retrieve_docs(query: str, k: int = 3) -> list:
    return [d.page_content for d in vectorstore.similarity_search(query, k=k)]


print(retrieve_docs("overpriced property Bengaluru"))


# ── 8. Retrieval Node (Conditional — Overpriced Only) ──────────────────────


def retrieval_node(state: AgentState) -> dict:
    if state["price_category"] != "Overpriced":
        return {"retrieved_docs": []}

    prop = state["property_data"]
    query = (
        f"Overpriced property in Bengaluru. "
        f"Location: {prop['location']}. "
        f"Area: {prop['total_sqft']} sqft. "
        f"BHK: {prop['bhk']}. "
        f"Price: ₹{state['predicted_price']:.2f} Lakhs."
    )
    return {"retrieved_docs": retrieve_docs(query)}


state.update(out)
print(retrieval_node(state))


# ── 9. Advisory Node (Final Structured Report) ──────────────────────────────

ACTION_MAP = {
    "Overpriced": (
        "CAUTION",
        "Exercise caution — negotiate hard or consider alternatives",
    ),
    "Fair Value": (
        "HOLD / CONSIDER",
        "Fair deal — proceed based on your needs and preferences",
    ),
    "Undervalued": ("BUY", "Strong investment opportunity — act promptly"),
}


def advisory_node(state: AgentState) -> dict:
    prop = state["property_data"]
    predicted_price = state["predicted_price"]
    price_category = state["price_category"]
    retrieved_docs = state["retrieved_docs"]
    market_analysis = state["market_analysis"]

    comps = get_comparable_properties(prop["location"], prop["bhk"])
    signal, action_hint = ACTION_MAP[price_category]
    docs_text = (
        "\n".join(f"- {d}" for d in retrieved_docs)
        if retrieved_docs
        else "Standard market conditions apply."
    )

    prompt = f"""
You are an AI real estate investment advisor for Bengaluru.

Property: {prop['bhk']} BHK | {prop['total_sqft']} sqft | {prop['location']}
Predicted Price: ₹{predicted_price:.2f} Lakhs | Category: {price_category}
Signal: {signal} | Action: {action_hint}

Market Analysis:
{market_analysis}

Market Intelligence (RAG):
{docs_text}

Comparable {prop['bhk']} BHK in {prop['location']}:
Range: ₹{comps['min']}L – ₹{comps['max']}L | Avg: ₹{comps['avg']}L ({comps['count']} listings)

Generate structured advisory (max 150 words). Use EXACTLY this format:

PROPERTY SUMMARY:
[2 lines on valuation and market position]

COMPARABLE ANALYSIS:
[1-2 lines comparing with similar properties]

INVESTMENT ACTION: {signal}
[1-2 lines on recommended action with clear reasoning]

DISCLAIMER:
This is an AI-generated advisory for informational purposes only. Consult a licensed real estate professional before any investment decision.
"""

    response = llm.invoke(prompt)
    return {"advisory_report": response.content}


state.update(retrieval_node(state))
state.update(advisory_node(state))
print(state["advisory_report"])


# ── 10. LangGraph Agent Pipeline ────────────────────────────────────────────


def route_by_category(state: AgentState) -> str:
    return {
        "Overpriced": "overpriced",
        "Fair Value": "fair_value",
        "Undervalued": "undervalued",
    }.get(state["price_category"], "fair_value")


from langgraph.graph import StateGraph, START, END

builder = StateGraph(AgentState)

builder.add_node("Market Analysis", market_analysis_node)
builder.add_node("Retrieval (Chroma)", retrieval_node)
builder.add_node("Advisory", advisory_node)

builder.add_edge(START, "Market Analysis")

builder.add_conditional_edges(
    "Market Analysis",
    route_by_category,
    {
        "overpriced": "Retrieval (Chroma)",
        "fair_value": "Advisory",
        "undervalued": "Advisory",
    },
)

builder.add_edge("Retrieval (Chroma)", "Advisory")
builder.add_edge("Advisory", END)

graph = builder.compile()


# ── 11. Full Agent Execution ─────────────────────────────────────────────────

initial_state = {
    "property_data": sample,
    "predicted_price": 0.0,
    "price_category": "",
    "market_analysis": "",
    "retrieved_docs": [],
    "advisory_report": "",
}

result = graph.invoke(initial_state)

print("\n" + "=" * 60)
print("FINAL ADVISORY REPORT")
print("=" * 60)
print(result["advisory_report"])


"""
### Key Design Decisions

- ML model (sklearn Pipeline) runs deterministically — no LLM involvement in price prediction
- LLM is constrained to enhance, not override, the ML output
- RAG activated only for Overpriced properties — efficient use of retrieval
- Conditional routing adapts the pipeline to property valuation context
- Structured output format enforces consistency across all advisory reports
- FakeEmbeddings used for free-tier compatibility (no embedding API cost)
"""
