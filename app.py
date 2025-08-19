import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

import json

from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import pandas as pd
from io import StringIO

from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import TavilyClient

# ========== Setup ==========
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

llm_name = "gemini-1.5-flash"
model = ChatGoogleGenerativeAI(api_key=google_api_key, model=llm_name)

tavily = TavilyClient(api_key=tavily_api_key)
memory = SqliteSaver.from_conn_string(":memory:")


# ========== Agent State ==========
class AgentState(TypedDict):
    task: str
    competitors: List[str]
    csv_file: str
    financial_data: str
    analysis: str
    competitor_data: str
    comparison: str
    feedback: str
    report: str
    content: List[str]
    revision_number: int
    max_revisions: int


class Queries(BaseModel):
    queries: List[str]


# ========== Prompts ==========
GATHER_FINANCIALS_PROMPT = """You are an expert financial analyst. Gather the financial data for the given company. Provide detailed financial data."""
ANALYZE_DATA_PROMPT = """You are an expert financial analyst. Analyze the provided financial data and provide detailed insights and analysis."""
RESEARCH_COMPETITORS_PROMPT = """You are a researcher tasked with providing information about similar companies for performance comparison. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""
COMPETE_PERFORMANCE_PROMPT = """You are an expert financial analyst. Compare the financial performance of the given company with its competitors based on the provided data. 
**MAKE SURE TO INCLUDE THE NAMES OF THE COMPETITORS IN THE COMPARISON.**"""
FEEDBACK_PROMPT = """You are a reviewer. Provide detailed feedback and critique for the provided financial comparison report. Include any additional information or revisions needed."""
WRITE_REPORT_PROMPT = """You are a financial report writer. Write a comprehensive financial report based on the analysis, competitor research, comparison, and feedback provided."""
RESEARCH_CRITIQUE_PROMPT = """You are a researcher tasked with providing information to address the provided critique. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""


# ========== Nodes ==========
def gather_financials_node(state: AgentState):
    df = pd.read_csv(StringIO(state["csv_file"]))

    # Summarize instead of dumping everything
    summary = df.describe(include="all").to_string()
    head = df.head(5).to_string()

    combined_content = f"""
Task: {state['task']}

Here is the financial data summary:
{summary}

Here are the first 5 rows of data:
{head}
"""

    messages = [
        SystemMessage(content=GATHER_FINANCIALS_PROMPT),
        HumanMessage(content=combined_content),
    ]
    response = model.invoke(messages)
    return {"financial_data": response.content}


def analyze_data_node(state: AgentState):
    messages = [
        SystemMessage(content=ANALYZE_DATA_PROMPT),
        HumanMessage(content=f"Task: {state['task']}\n\n{state['financial_data']}"),
    ]
    response = model.invoke(messages)
    return {"analysis": response.content}


def research_competitors_node(state: AgentState):
    content = state.get("content", [])
    for competitor in state["competitors"]:
        queries = model.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=RESEARCH_COMPETITORS_PROMPT),
                HumanMessage(content=f"Task: {state['task']}\nCompetitor: {competitor}"),
            ]
        )
        for q in queries.queries:
            try:
                response = tavily.search(query=q, max_results=2)
                for r in response.get("results", []):
                    content.append(r["content"])
            except Exception as e:
                content.append(f"Error fetching data for {competitor}: {e}")
    return {"content": content}


def compare_performance_node(state: AgentState):
    content = "\n\n".join(state.get("content", []))
    user_message = HumanMessage(
        content=f"Task: {state['task']}\n\nHere is the financial analysis:\n\n{state['analysis']}"
    )
    messages = [
        SystemMessage(content=COMPETE_PERFORMANCE_PROMPT),
        user_message,
        HumanMessage(content=f"Additional competitor data:\n{content}"),
    ]
    response = model.invoke(messages)
    return {
        "comparison": response.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }


def collect_feedback_node(state: AgentState):
    messages = [
        SystemMessage(content=FEEDBACK_PROMPT),
        HumanMessage(content=f"Task: {state['task']}\n\n{state['comparison']}"),
    ]
    response = model.invoke(messages)
    return {"feedback": response.content}


def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=f"Task: {state['task']}\n\nFeedback: {state['feedback']}"),
        ]
    )
    content = state.get("content", [])
    for q in queries.queries:
        try:
            response = tavily.search(query=q, max_results=2)
            for r in response.get("results", []):
                content.append(r["content"])
        except Exception as e:
            content.append(f"Error fetching critique research: {e}")
    return {"content": content}


def write_report_node(state: AgentState):
    messages = [
        SystemMessage(content=WRITE_REPORT_PROMPT),
        HumanMessage(content=f"Task: {state['task']}\n\n{state['comparison']}"),
    ]
    response = model.invoke(messages)
    return {"report": response.content}


# ========== Graph Logic ==========
def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return "write_report"
    return "collect_feedback"


builder = StateGraph(AgentState)

builder.add_node("gather_financials", gather_financials_node)
builder.add_node("analyze_data", analyze_data_node)
builder.add_node("research_competitors", research_competitors_node)
builder.add_node("compare_performance", compare_performance_node)
builder.add_node("collect_feedback", collect_feedback_node)
builder.add_node("research_critique", research_critique_node)
builder.add_node("write_report", write_report_node)

builder.set_entry_point("gather_financials")

builder.add_edge("gather_financials", "analyze_data")
builder.add_edge("analyze_data", "research_competitors")
builder.add_edge("research_competitors", "compare_performance")

builder.add_conditional_edges(
    "compare_performance",
    should_continue,
    {"collect_feedback": "collect_feedback", "write_report": "write_report"},
)

builder.add_edge("collect_feedback", "research_critique")
builder.add_edge("research_critique", "compare_performance")

graph = builder.compile(checkpointer=memory)


# ========== Streamlit UI ==========
import streamlit as st

def main():
    st.title("📊 Financial Performance Reporting Agent")

    task = st.text_input(
        "Enter the task:",
        "Analyze the financial performance of our company (MyAICo.AI) compared to competitors",
    )
    competitors = st.text_area("Enter competitor names (one per line):").split("\n")
    max_revisions = st.number_input("Max Revisions", min_value=1, value=2)
    uploaded_file = st.file_uploader(
        "Upload a CSV file with the company's financial data", type=["csv"]
    )

    if st.button("Start Analysis") and uploaded_file is not None:
        csv_data = uploaded_file.getvalue().decode("utf-8")

        initial_state = {
            "task": task,
            "competitors": [comp.strip() for comp in competitors if comp.strip()],
            "csv_file": csv_data,
            "max_revisions": max_revisions,
            "revision_number": 1,
        }
        thread = {"configurable": {"thread_id": "1"}}

        final_state = None
        for s in graph.stream(initial_state, thread):
            st.write(s)
            final_state = s

        if final_state and "report" in final_state:
            st.subheader("📑 Final Report")
            st.write(final_state["report"])


if __name__ == "__main__":
    main()
