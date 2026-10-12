"""
app.py -- Streamlit chat frontend for the Olist Text-to-SQL agent.

Run with:  streamlit run app.py
"""

import streamlit as st
from graph import run_agent
from schema_context import build_schema_context

st.set_page_config(page_title="Olist Text-to-SQL Agent", page_icon="🛒", layout="centered")


@st.cache_resource(show_spinner="Reading database schema...")
def load_schema() -> tuple[str | None, str | None]:
    """Built once per app session; returns (schema_context, error_message)."""
    try:
        return build_schema_context(), None
    except Exception as e:
        return None, str(e)


schema_context, db_error = load_schema()

# ---- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.title("🛒 Olist SQL Agent")
    st.caption("Ask questions about the Brazilian E-Commerce dataset in plain English.")
    max_retries = st.slider("Max self-correction retries", 0, 5, 3)
    show_sql = st.checkbox("Show generated SQL", value=True)
    st.markdown("**Try asking:**")
    examples = [
        "How many orders were delivered?",
        "Top 5 states by number of customers",
        "Which 5 product categories have the highest average review score?",
        "What is the average freight value per product category?",
        "What are the most used payment types?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["pending_question"] = ex
    if st.button("Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ---- Check Database Status ----------------------------------------------
if not schema_context:
    st.error(f"⚠️ Could not connect to MySQL database: {db_error}")
    st.info("Please ensure MySQL is running on localhost:3306 and run `python py_script.py` to import the datasets.")
    if st.button("🔄 Retry Connection"):
        st.cache_resource.clear()
        st.rerun()
    st.stop()

# ---- Chat history --------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sql and msg.get("sql"):
            with st.expander(f"SQL used (retries: {msg.get('retries', 0)})"):
                st.code(msg["sql"], language="sql")

# ---- Input ---------------------------------------------------------------
typed = st.chat_input("Ask a question about the data...")
question = typed or st.session_state.pop("pending_question", None)

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking, querying, and checking the result..."):
            try:
                result = run_agent(question, schema_context, max_retries=max_retries)
                answer = result["final_answer"]
                sql = result.get("generated_sql")
                retries = result.get("retry_count", 0)
            except Exception as e:
                answer = f"Something went wrong while running the agent: {e}"
                sql, retries = None, 0

        st.markdown(answer)
        if show_sql and sql:
            with st.expander(f"SQL used (retries: {retries})"):
                st.code(sql, language="sql")

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "sql": sql, "retries": retries}
    )
