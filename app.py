import streamlit as st
import os
import pickle
import pandas as pd
from typing import Dict, List, TypedDict

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langgraph.graph import StateGraph, START, END

st.set_page_config(
    page_title="Real Estate AI Advisory Agent",
    page_icon="🏠",
    layout="wide"
)

# ── Cached Resource Setup ────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    with open("price_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def setup_llm():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to .env or Streamlit secrets.")
        st.stop()
    return ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")


@st.cache_resource
def load_data():
    data = pd.read_csv("Bengaluru_House_Data.csv")
    data = data.drop(['society', 'availability'], axis=1, errors='ignore')
    data = data.dropna()

    def convert_sqft(x):
        try:
            if '-' in str(x):
                a, b = x.split('-')
                return (float(a) + float(b)) / 2
            return float(x)
        except:
            return None

    data['total_sqft'] = data['total_sqft'].apply(convert_sqft)
    data = data.dropna()

    extracted = data['size'].str.split().str[0]
    data['bhk'] = pd.to_numeric(extracted, errors='coerce')
    data.dropna(subset=['bhk'], inplace=True)
    data['bhk'] = data['bhk'].astype(int)
    data = data.drop('size', axis=1)
    data['price_per_sqft'] = (data['price'] * 100000) / data['total_sqft']
    return data


@st.cache_resource
def setup_rag():
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
        "A property priced more than 20% above the location average per-sqft rate is a high negotiation risk."
    ]
    return Chroma.from_texts(documents, FakeEmbeddings(size=384))


model       = load_model()
llm         = setup_llm()
data        = load_data()
vectorstore = setup_rag()

location_avg_ppsf = data.groupby('location')['price_per_sqft'].mean().to_dict()
global_avg_ppsf   = float(data['price_per_sqft'].mean())


# ── Agent State & Logic ──────────────────────────────────────────────────────

class AgentState(TypedDict):
    property_data:   Dict
    predicted_price: float
    price_category:  str
    market_analysis: str
    retrieved_docs:  List[str]
    advisory_report: str


def predict_price(prop: dict) -> float:
    return float(model.predict(pd.DataFrame([prop]))[0])


def get_price_category(location: str, total_sqft: float, predicted_price: float) -> str:
    ppsf     = (predicted_price * 100000) / total_sqft
    avg_ppsf = location_avg_ppsf.get(location, global_avg_ppsf)
    ratio    = ppsf / avg_ppsf
    if ratio > 1.20:
        return "Overpriced"
    elif ratio < 0.80:
        return "Undervalued"
    return "Fair Value"


def get_comparable_properties(location: str, bhk: int) -> dict:
    comps = data[(data['location'] == location) & (data['bhk'] == bhk)]['price']
    if len(comps) < 3:
        comps = data[data['bhk'] == bhk]['price']
    return {
        "min":   round(float(comps.min()), 2),
        "max":   round(float(comps.max()), 2),
        "avg":   round(float(comps.mean()), 2),
        "count": int(len(comps))
    }


def retrieve_docs(query: str, k: int = 3) -> list:
    return [d.page_content for d in vectorstore.similarity_search(query, k=k)]


ACTION_MAP = {
    "Overpriced":  ("CAUTION",         "Exercise caution — negotiate hard or consider alternatives"),
    "Fair Value":  ("HOLD / CONSIDER", "Fair deal — proceed based on your needs and preferences"),
    "Undervalued": ("BUY",             "Strong investment opportunity — act promptly"),
}


def market_analysis_node(state: AgentState) -> dict:
    prop            = state["property_data"]
    predicted_price = predict_price(prop)
    price_category  = get_price_category(prop["location"], prop["total_sqft"], predicted_price)

    ppsf     = (predicted_price * 100000) / prop["total_sqft"]
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
- Do NOT add assumptions beyond given data
- Max 80 words

Output EXACTLY in this format:
Market Position:
Key Value Drivers:
"""

    response = llm.invoke(prompt)
    return {
        "predicted_price": predicted_price,
        "price_category":  price_category,
        "market_analysis": response.content
    }


def retrieval_node(state: AgentState) -> dict:
    if state["price_category"] != "Overpriced":
        return {"retrieved_docs": []}
    prop  = state["property_data"]
    query = (
        f"Overpriced property in Bengaluru. "
        f"Location: {prop['location']}. "
        f"Area: {prop['total_sqft']} sqft. "
        f"BHK: {prop['bhk']}. "
        f"Price: ₹{state['predicted_price']:.2f} Lakhs."
    )
    return {"retrieved_docs": retrieve_docs(query)}


def advisory_node(state: AgentState) -> dict:
    prop            = state["property_data"]
    predicted_price = state["predicted_price"]
    price_category  = state["price_category"]
    retrieved_docs  = state["retrieved_docs"]
    market_analysis = state["market_analysis"]

    comps               = get_comparable_properties(prop["location"], prop["bhk"])
    signal, action_hint = ACTION_MAP[price_category]
    docs_text           = "\n".join(f"- {d}" for d in retrieved_docs) if retrieved_docs else "Standard market conditions apply."

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


def route_by_category(state: AgentState) -> str:
    return {
        "Overpriced":  "overpriced",
        "Fair Value":  "fair_value",
        "Undervalued": "undervalued"
    }.get(state["price_category"], "fair_value")


# Build LangGraph
builder = StateGraph(AgentState)
builder.add_node("Market Analysis",    market_analysis_node)
builder.add_node("Retrieval (Chroma)", retrieval_node)
builder.add_node("Advisory",           advisory_node)
builder.add_edge(START, "Market Analysis")
builder.add_conditional_edges(
    "Market Analysis",
    route_by_category,
    {
        "overpriced":  "Retrieval (Chroma)",
        "fair_value":  "Advisory",
        "undervalued": "Advisory"
    }
)
builder.add_edge("Retrieval (Chroma)", "Advisory")
builder.add_edge("Advisory", END)
graph = builder.compile()


# ── Streamlit UI ─────────────────────────────────────────────────────────────

st.title("🏠 Real Estate AI Advisory Agent")
st.markdown("### Hybrid ML + LLM System for Bengaluru Property Investment")
st.caption("Predict property price and get AI-powered investment advisory — powered by LangGraph + RAG.")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    area_type  = st.selectbox("🏗️ Area Type", sorted(data["area_type"].unique()))
    total_sqft = st.number_input("📐 Total Square Feet", min_value=300, max_value=10000, value=1200)
    bath       = st.number_input("🚿 Bathrooms", min_value=1, max_value=10, value=2)

with col2:
    location = st.selectbox("📍 Location", sorted(data["location"].dropna().unique()))
    bhk      = st.number_input("🛏️ BHK", min_value=1, max_value=10, value=2)
    balcony  = st.number_input("🌿 Balconies", min_value=0, max_value=5, value=1)

with col3:
    st.markdown("### 📊 Dataset Stats")
    st.metric("Total Listings", f"{len(data):,}")
    st.metric("Unique Locations", f"{data['location'].nunique():,}")
    st.metric("Avg Price", f"₹{round(data['price'].mean(), 1)} L")

st.markdown("---")

if st.button("🔮 Get AI Advisory", type="primary", use_container_width=True):

    property_data = {
        "area_type":  area_type,
        "location":   location,
        "total_sqft": float(total_sqft),
        "bath":       float(bath),
        "balcony":    float(balcony),
        "bhk":        int(bhk)
    }

    with st.spinner("🤖 Running AI Agent Pipeline... Please wait"):
        initial_state = {
            "property_data":   property_data,
            "predicted_price": 0.0,
            "price_category":  "",
            "market_analysis": "",
            "retrieved_docs":  [],
            "advisory_report": ""
        }
        result = graph.invoke(initial_state)

    price    = result["predicted_price"]
    category = result["price_category"]

    if category == "Overpriced":
        badge, label = "🔴", "OVERPRICED"
    elif category == "Undervalued":
        badge, label = "🟢", "UNDERVALUED — OPPORTUNITY"
    else:
        badge, label = "🟡", "FAIR VALUE"

    st.markdown(f"## {badge} {label}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Price", f"₹{price:.2f} Lakhs")
    m2.metric("Your Price / sqft", f"₹{(price * 100000 / total_sqft):,.0f}")
    m3.metric("Location Avg / sqft", f"₹{location_avg_ppsf.get(location, global_avg_ppsf):,.0f}")

    if result["retrieved_docs"]:
        st.success(f"📚 RAG Activated — {len(result['retrieved_docs'])} market guidelines retrieved")
    else:
        st.info("⚡ Fast Path — Direct advisory (retrieval skipped)")

    st.markdown("---")

    st.markdown("### 🔍 Market Analysis")
    st.info(result["market_analysis"])

    st.markdown("### 💡 Investment Advisory Report")
    st.success(result["advisory_report"])

    if result["retrieved_docs"]:
        st.markdown("### 📚 Retrieved Market Intelligence")
        for doc in result["retrieved_docs"]:
            st.markdown(f"- {doc}")

    st.markdown("---")

else:
    st.markdown("""
    <div style="text-align:center; padding:3rem; color:#888;">
        <p style="font-size:3rem;">👆</p>
        <p style="font-size:1.2rem;">Configure property details above and click <b>Get AI Advisory</b></p>
    </div>
    """, unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.markdown("## 📊 System Info")
    st.markdown("""
    **ML Model:** Linear Regression (R² = 0.83)
    **LLM:** Groq LLaMA 3.3 70B
    **RAG:** Chroma Vectorstore
    **Framework:** LangGraph
    **Dataset:** 13,000+ Bengaluru listings

    ---

    ## 🔧 Agent Pipeline
    - ✅ ML Price Prediction
    - ✅ LLM Market Analysis
    - ✅ Conditional RAG Routing
    - ✅ Investment Advisory

    ---

    ## 📈 Price Categories
    - **🔴 Overpriced:** >20% above avg/sqft
    - **🟡 Fair Value:** within ±20%
    - **🟢 Undervalued:** <20% below avg/sqft
    """)
    st.markdown("---")
    st.markdown("Built for GenAI End-Sem Project — Milestone 2")
