# ============================================================
# Page 5 — Scenario Q&A Frontend (Streamlit)
# ============================================================
# This page lets users ask questions like:
# "What will happen if the market falls and we raise interest rates?
#  How will it affect our target variable?"
#
# The actual scenario calculation / simulation is handled by a backend
# (BigQuery, Airflow, Cloud Function, etc.). This page is just the UI.
# ============================================================

import os
import requests
import streamlit as st

# ------------------------------------------------------------
# CONFIG — Backend endpoint for scenario engine
# ------------------------------------------------------------
# Prefer environment variable so you can change without editing code
SCENARIO_API_URL = os.getenv(
    "SCENARIO_API_URL",
    # TODO: replace this with your real Cloud Function / API URL
    "https://us-central1-your-project-id.cloudfunctions.net/credit_scenario_engine"
)

# ------------------------------------------------------------
# Streamlit Setup
# ------------------------------------------------------------

st.title("Scenario Q&A — What If Analysis 🤖")
st.subheader("Ask natural language questions about delinquency under different market conditions")

# Sidebar
st.sidebar.header("About this page")
st.sidebar.markdown(
    """
This page is a **front-end only** interface.

- You ask *"what-if"* questions in natural language.
- The question is sent to a **backend scenario engine** (BigQuery / Airflow / Cloud Function).
- The backend returns a **textual explanation** of the expected impact on the target variable.

Examples:
- *"If unemployment increases by 2% and interest rates rise by 50 bps, how does delinquency change?"*
- *"What happens to our delinquency forecast if GDP growth drops to 0%?"*
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Backend URL in use:**")
st.sidebar.code(SCENARIO_API_URL, language="text")

# ------------------------------------------------------------
# Session State — store chat history for this page
# ------------------------------------------------------------
if "scenario_messages" not in st.session_state:
    st.session_state.scenario_messages = []

# ------------------------------------------------------------
# Backend Call Helper
# ------------------------------------------------------------
def call_scenario_engine(question: str) -> str:
    """
    Call the backend scenario engine with the user's question.
    The backend is expected to return JSON like:
        {"question": "...", "answer": "..."}
    You can adjust this to match your real API contract.
    """
    payload = {
        "query": question,        # main user question
        # You can send extra context if needed:
        # "context": {"user_role": "analyst", "project": "credit_delinquency"}
    }
    resp = requests.post(SCENARIO_API_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Fallback to empty string if key is missing
    return data.get("answer", "") or data.get("result", "") or "No answer returned from backend."

# ------------------------------------------------------------
# Main Chat UI
# ------------------------------------------------------------
st.markdown("---")
st.markdown("### Ask a scenario question")

with st.expander("💡 Example questions", expanded=False):
    st.markdown(
        """
- *"If the stock market falls by 10% and we raise interest rates by 0.5%, how will it impact delinquency next quarter?"*  
- *"What if unemployment goes above 6% but we keep rates constant — how sensitive is delinquency?"*  
- *"How do lower GDP growth and higher inflation together affect our delinquency forecasts?"*
        """
    )

# Replay past conversation
for message in st.session_state.scenario_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input (bottom of the page)
prompt = st.chat_input("Describe a scenario (e.g., market fall + higher interest rates)…")

if prompt:
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user message to history
    st.session_state.scenario_messages.append(
        {"role": "user", "content": prompt}
    )

    # Call backend and display answer
    try:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing scenario with backend engine…"):
                answer = call_scenario_engine(prompt)
                st.markdown(answer)

        # Save assistant message
        st.session_state.scenario_messages.append(
            {"role": "assistant", "content": answer}
        )

    except Exception as e:
        error_msg = f"❌ Error calling scenario engine: {e}"
        with st.chat_message("assistant"):
            st.error(error_msg)
        st.session_state.scenario_messages.append(
            {"role": "assistant", "content": error_msg}
        )
