"""
graph.py

This is where every piece built so far gets wired together into one
LangGraph state machine.

Flow:
  generate_sql -> validate_sql -> (if invalid) loop back to generate_sql,
  with the error fed back in, up to max_retries
                              -> (if valid) execute_sql -> (if DB error) loop
  back to generate_sql, same retry logic
                                                        -> (if success)
  synthesize_response -> END

  If retries are exhausted at either point -> give_up -> END, with an
  honest "couldn't answer this" message instead of a fake/broken answer.
"""

from langgraph.graph import StateGraph, END

from state import AgentState
from generate_sql import generate_sql
from validate_sql import validate_sql
from execute_sql import execute_sql
from synthesize_response import synthesize_response, build_failure_message


# ---- Node functions -----------------------------------------------------
# Each one takes the full state and returns only the fields it changes.
# LangGraph merges the returned dict into the running state.

def generate_sql_node(state: AgentState) -> dict:
    previous_sql = state.get("generated_sql")

    execution_result = state.get("execution_result")
    execution_error = execution_result["error"] if execution_result and not execution_result["success"] else None
    previous_error = state.get("validation_error") or execution_error

    is_retry = previous_error is not None
    new_retry_count = state.get("retry_count", 0) + (1 if is_retry else 0)

    sql = generate_sql(
        question=state["question"],
        schema_context=state["schema_context"],
        previous_sql=previous_sql if is_retry else None,
        previous_error=previous_error,
    )

    # Clear old error/result fields so the routing functions downstream
    # always see fresh results from THIS attempt, not a stale one.
    return {
        "generated_sql": sql,
        "retry_count": new_retry_count,
        "validation_error": None,
        "execution_result": None,
    }


def validate_sql_node(state: AgentState) -> dict:
    is_valid, message = validate_sql(state["generated_sql"])
    return {"validation_error": None if is_valid else message}


def execute_sql_node(state: AgentState) -> dict:
    result = execute_sql(state["generated_sql"])
    return {"execution_result": result}


def synthesize_response_node(state: AgentState) -> dict:
    answer = synthesize_response(state["question"], state["execution_result"])
    return {"final_answer": answer}


def give_up_node(state: AgentState) -> dict:
    execution_result = state.get("execution_result")
    last_error = (
        state.get("validation_error")
        or (execution_result["error"] if execution_result else "Unknown error")
    )
    answer = build_failure_message(state["question"], last_error)
    return {"final_answer": answer}


# ---- Routing functions ---------------------------------------------------
# These decide which node runs next, based on the current state.

def route_after_validation(state: AgentState) -> str:
    if state.get("validation_error"):
        if state["retry_count"] >= state["max_retries"]:
            return "give_up"
        return "generate_sql"
    return "execute_sql"


def route_after_execution(state: AgentState) -> str:
    execution_result = state.get("execution_result")
    if execution_result and not execution_result["success"]:
        if state["retry_count"] >= state["max_retries"]:
            return "give_up"
        return "generate_sql"
    return "synthesize_response"


# ---- Build the graph -------------------------------------------------

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("validate_sql", validate_sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("synthesize_response", synthesize_response_node)
    workflow.add_node("give_up", give_up_node)

    workflow.set_entry_point("generate_sql")

    workflow.add_edge("generate_sql", "validate_sql")

    workflow.add_conditional_edges(
        "validate_sql",
        route_after_validation,
        {"execute_sql": "execute_sql", "generate_sql": "generate_sql", "give_up": "give_up"},
    )

    workflow.add_conditional_edges(
        "execute_sql",
        route_after_execution,
        {"synthesize_response": "synthesize_response", "generate_sql": "generate_sql", "give_up": "give_up"},
    )

    workflow.add_edge("synthesize_response", END)
    workflow.add_edge("give_up", END)

    return workflow.compile()


def run_agent(question: str, schema_context: str, max_retries: int = 3) -> dict:
    """
    Convenience wrapper -- this is what your Streamlit app will call.
    Returns the full final state, so you can pull out final_answer,
    generated_sql (useful for a "show SQL" toggle), retry_count, etc.
    """
    app = build_graph()

    initial_state: AgentState = {
        "question": question,
        "schema_context": schema_context,
        "generated_sql": None,
        "validation_error": None,
        "execution_result": None,
        "retry_count": 0,
        "max_retries": max_retries,
        "final_answer": None,
    }

    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    # Full end-to-end test against your REAL database and REAL LLM.
    # Run `python graph.py` to try it.
    from schema_context import build_schema_context

    real_schema = build_schema_context()

    question = "Which 5 product categories have the highest average review score?"
    print(f"Question: {question}\n")

    result = run_agent(question, real_schema)

    print("Generated SQL (final attempt):")
    print(result["generated_sql"])
    print(f"\nRetries used: {result['retry_count']}")
    print("\nAnswer:")
    print(result["final_answer"])
