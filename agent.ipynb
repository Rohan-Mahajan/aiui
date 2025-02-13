from google.colab import userdata
import pandas as pd
import numpy as np
import logging
import random
import threading
import sys
import time
import re
from typing import TypedDict, List

# import smtplib
# from email.message import EmailMessage
# import os

# Import LangGraph and related components
from langgraph.graph import END, StateGraph
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq
from langchain.chat_models import AzureChatOpenAI


# def send_email(reciepient, subject, content, sender, password):
#   msg = EmailMessage()
#   msg.set_content(content)
#   msg['Subject'] = subject
#   msg['From'] = sender
#   msg['To'] = reciepient

#   with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
#     smtp.login(sender, password)
#     smtp.send_message(msg)

azure_api_key = "Azure_api_key"
azure_endpoint = "azure_endpoint"
deployment_name = "llm_model"

llm = AzureChatOpenAI(
    deployment_name=deployment_name,
    openai_api_key=azure_api_key,
    openai_api_base=azure_endpoint,
    openai_api_version="version",
    model_name = "model_name",
    temperature = 0.3
)


# defining a similarity function
from difflib import SequenceMatcher

def similarity(a: str, b:str) -> float:
  return SequenceMatcher(None, a, b).ratio()

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Data Loading and Document Preparation
# groq_api = userdata.get("groq_api_key")
# Load defects file 
df = pd.read_csv("/content/defects.csv")

# Load the test cases CSV.
try:
    test_cases_df = pd.read_csv("/content/test_cases.csv")
except Exception as e:
    logging.warning("Test cases file not found or unreadable. Creating an empty DataFrame.")
    test_cases_df = pd.DataFrame(columns=["Module", "Test_Scenario", "Test_Steps", "Pre_Requisite", "Pass_Fail_Criteria", "Expected_Result"])

# Build documents from defects CSV (only using Module, Description, Solution)
docs = []
for _, row in df.iterrows():
    if pd.notna(row["Description"]) and pd.notna(row["Solution"]):
        docs.append(Document(
            page_content=row["Description"],
            metadata={
                "solution": row["Solution"],
                "module": row["Module"]
            }
        ))

# Create embeddings and FAISS vector store for retrieval
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# Helper: Classify Test Case (Positive/Negative) based on keywords
def classify_test_case(tc_text: str) -> str:
    negative_keywords = [
        "fail", "error", "misconfigured", "incorrect", "doesn't work",
        "not work", "negative", "invalid", "wrong", "missing", "unexpected",
        "should not", "incorrectly", "failure", "reject", "malformed",
        "timeout", "invalid input", "edge case", "out of bounds"
    ]
    text_lower = tc_text.lower()
    return "negative" if any(kw in text_lower for kw in negative_keywords) else "positive"

# Helper: Retrieve CSV Test Cases by Module
def get_csv_test_cases(module: str):
    """
    Returns a tuple of two lists: (positive_test_cases, negative_test_cases).
    Each test case is constructed from the CSV columns and validated.
    """
    positive_cases = []
    negative_cases = []
    
    global test_cases_df
    if test_cases_df.empty:
        return positive_cases, negative_cases

    # Filter by Module
    df_filtered = test_cases_df[test_cases_df["Module"] == module]
    
    for _, row in df_filtered.iterrows():
        # Combine the five columns into one text block for classification
        tc_text = " ".join([
            str(row.get("Test_Scenario", "")),
            str(row.get("Test_Steps", "")),
            str(row.get("Pre_Requisite", "")),
            str(row.get("Expected_Result", "")),
            str(row.get("Pass_Fail_Criteria", ""))
        ])
        if tc_text.strip() == "":
            continue
        tc_type = classify_test_case(tc_text)
        # Also store the row as a dictionary for later formatting
        tc_dict = {
            "Module": row["Module"],
            "Test_Scenario": row["Test_Scenario"],
            "Test_Steps": row["Test_Steps"],
            "Pre_Requisite": row["Pre_Requisite"],
            "Pass_Fail_Criteria": row["Pass_Fail_Criteria"],
            "Expected_Result": row["Expected_Result"]
        }
        if tc_type == "positive":
            positive_cases.append(tc_dict)
        else:
            negative_cases.append(tc_dict)
    return positive_cases, negative_cases

# Helper: Parse Generated Test Case into Fields
def parse_test_case(tc_text: str) -> dict:
    """
    Parses a generated test case text to extract the fields.
    Expected labels: Test_Scenario:, Test_Steps:, Pre_Requisite:, Expected_Result:, Pass_Fail_Criteria:
    """
    fields = {
        "Test_Scenario": "",
        "Test_Steps": "",
        "Pre_Requisite": "",
        "Expected_Result": "",
        "Pass_Fail_Criteria": ""
    }
    # Use regex to capture content after each label up to the next label or end.
    for field in fields.keys():
        # Pattern looks for e.g. "Test_Scenario:" then capture until the next field label
        pattern = field + r":\s*(.*?)\s*(?=(Test_Steps:|Pre_Requisite:|Expected_Result:|Pass_Fail_Criteria:|$))"
        match = re.search(pattern, tc_text, re.DOTALL)
        if match:
            fields[field] = match.group(1).strip()
    return fields

# Helper: Save New Test Cases to CSV (avoiding duplicates)
def save_new_test_cases(new_cases: List[dict]):
    """
    new_cases: list of dictionaries with keys:
      Module, Test_Scenario, Test_Steps, Pre_Requisite, Pass_Fail_Criteria, Expected_Result.
    Appends only new (non-duplicate) test cases to the CSV.
    """
    global test_cases_df
    required_columns = ["Module", "Test_Scenario", "Test_Steps", "Pre_Requisite", "Pass_Fail_Criteria", "Expected_Result"]
    if test_cases_df.empty:
        test_cases_df = pd.DataFrame(columns=required_columns)
    
    rows_to_add = []
    for case in new_cases:
        # Consider a test case duplicate if all fields match for the same module.
        duplicate = test_cases_df[
            (test_cases_df["Module"] == case["Module"]) &
            (test_cases_df["Test_Scenario"] == case["Test_Scenario"]) &
            (test_cases_df["Test_Steps"] == case["Test_Steps"]) &
            (test_cases_df["Pre_Requisite"] == case["Pre_Requisite"]) &
            (test_cases_df["Pass_Fail_Criteria"] == case["Pass_Fail_Criteria"]) &
            (test_cases_df["Expected_Result"] == case["Expected_Result"])
        ]
        if duplicate.empty:
            rows_to_add.append(case)
    if rows_to_add:
        new_df = pd.DataFrame(rows_to_add)
        test_cases_df = pd.concat([test_cases_df, new_df], ignore_index=True)
        test_cases_df.to_csv("/content/test_cases.csv", index=False)
        logging.info("Saved %d new test case(s) to CSV.", len(rows_to_add))
    else:
        logging.info("No new test cases to save (duplicates skipped).")

# Define Agent State and LLM Initialization
class AgentState(TypedDict):
    input: str
    context: List[Document]
    response: str

# llm = ChatGroq(
#     groq_api_key=groq_api,
#     temperature=0.3,
#     model_name="gemma2-9b-it",
# )

# Workflow Node: Validate or Generate Test Cases (with CSV storage)

def validate_or_generate_test_cases(state: AgentState):
    try:
        if not state["context"]:
            return {"response": "**Error**: The defect could not be found in the database."}
        context = state["context"][0]
        error_message = state["input"]
        # Ensure the defect is similar to the error message
        if similarity(error_message.lower(), context.page_content.lower()) < 0.3:
            return {"response": "**Error**: The defect could not be found in the database."}
        solution = context.metadata["solution"]
        module = context.metadata["module"]

        # Generate explanation for why the solution works
        explanation_prompt = """
        [INST] Explain why this solution fixes the following error:
        Error: {error}
        Solution: {solution}
        [/INST]
        """
        explanation_template = ChatPromptTemplate.from_template(explanation_prompt)
        formatted_explanation = explanation_template.format_prompt(error=error_message, solution=solution)
        explanation = llm.invoke(formatted_explanation.to_messages()).content.strip()

        # New test case generation prompt without fixed counts.
        test_case_prompt = """
        [INST] Generate a comprehensive set of test cases to fully validate the following defect solution end-to-end.
        Error: {error}
        Solution: {solution}
        Explanation: {explanation}
        Your task is to cover all scenarios in which the solution should work (positive cases) and also those where it might fail (negative cases) if applicable.
        Each test case MUST include the following sections and end with the delimiter "### END TEST CASE ###":
        Test_Scenario: A short description of the scenario.
        Test_Steps: Step-by-step instructions.
        Pre_Requisite: Conditions before running the test.
        Expected_Result: What should happen.
        Pass_Fail_Criteria: How to determine if the test passes.
        
        Output format (including the delimiter):
        1. Test_Scenario: 
           Test_Steps: 
           Pre_Requisite: 
           Expected_Result: 
           Pass_Fail_Criteria: 
        ### END TEST CASE ###
        [/INST]
        """
        test_case_template = ChatPromptTemplate.from_template(test_case_prompt)
        formatted_test_cases = test_case_template.format_prompt(
            error=error_message, solution=solution, explanation=explanation)
        test_cases_response = llm.invoke(formatted_test_cases.to_messages()).content.strip()

        # Split the response using the delimiter
        delimiter = "\n### END TEST CASE ###\n"
        test_cases_raw = [tc.strip() for tc in re.split(delimiter, test_cases_response) if tc.strip()]

        generated_cases = []
        for tc_raw in test_cases_raw:
            parsed = parse_test_case(tc_raw)
            if parsed["Test_Scenario"]: # Ensure that parsing was successful
                generated_cases.append({
                    "Module": module,
                    "Test_Scenario": parsed["Test_Scenario"],
                    "Test_Steps": parsed["Test_Steps"],
                    "Pre_Requisite": parsed["Pre_Requisite"],
                    "Pass_Fail_Criteria": parsed["Pass_Fail_Criteria"],
                    "Expected_Result": parsed["Expected_Result"]
                })

        # Save any new generated test cases to the CSV (avoiding duplicates)
        if generated_cases:
            save_new_test_cases(generated_cases)

        # Format the test cases for the final output
        def format_tc(tc: dict) -> str:
            return (f"Test_Scenario: {tc['Test_Scenario']}\n"
                    f"Test_Steps: {tc['Test_Steps']}\n"
                    f"Pre_Requisite: {tc['Pre_Requisite']}\n"
                    f"Expected_Result: {tc['Expected_Result']}\n"
                    f"Pass_Fail_Criteria: {tc['Pass_Fail_Criteria']}")

        test_cases_text = "\n\n".join(format_tc(tc) for tc in generated_cases)
        response_template = (
            "**Error:**\n{Error}\n\n"
            "**Solution:**\n{Solution}\n\n"
            "**Explanation:**\n{Explanation}\n\n"
            "**Test Cases:**\n{TestCases}"
        )
        return {"response": response_template.format(
            Error=error_message,
            Solution=solution,
            Explanation=explanation,
            TestCases=test_cases_text
        )}

    except Exception as e:
        logging.error("Validation/Generation error: %s", str(e))
        return {"response": f"Error processing request: {str(e)}"}

# Build the State Graph Workflow
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", lambda state: {"context": retriever.invoke(state["input"])})
workflow.add_node("validate_or_generate_test_cases", validate_or_generate_test_cases)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "validate_or_generate_test_cases")
workflow.add_edge("validate_or_generate_test_cases", END)
agent = workflow.compile()

# Automated Evaluation & Self-improvement Functions
def auto_evaluate_solution(response: str) -> int:
    if "### END TEST CASE ###" in response:
        return 5
    elif "**Error**:" in response:
        return 1
    else:
        return 3

def generate_alternative_solution(error_message: str) -> str:
    alt_prompt = """
    [INST] Provide a concise, actionable alternative solution for the following error:
    Error: {error}
    Ensure that the solution is clear and does not include any follow-up questions.
    [/INST]
    """
    alt_template = ChatPromptTemplate.from_template(alt_prompt)
    formatted_alt = alt_template.format_prompt(error=error_message)
    alternative_solution = llm.invoke(formatted_alt.to_messages()).content.strip()

    test_case_prompt = """
    [INST] Given the error and the alternative solution:
    Error: {error}
    Solution: {solution}
    Generate EXACTLY 4 structured test cases (2 positive and 2 negative) with the delimiter "### END TEST CASE ###" after each test case.
    Each test case must include:
      Test_Scenario
      Test_Steps
      Pre_Requisite
      Expected_Result
      Pass_Fail_Criteria
    [/INST]
    """
    tc_template = ChatPromptTemplate.from_template(test_case_prompt)
    formatted_tc = tc_template.format_prompt(error=error_message, solution=alternative_solution)
    alternative_test_cases = llm.invoke(formatted_tc.to_messages()).content.strip()

    alt_response = (
        "**Alternative Solution (Generated):**\n{AltSolution}\n\n"
        "**Test Cases for Alternative Solution:**\n{AltTestCases}"
    ).format(
        AltSolution=alternative_solution,
        AltTestCases=alternative_test_cases
    )
    return alt_response

def get_solution_autonomously(error_message: str) -> str:
    max_iterations = 3
    iteration = 0
    while iteration < max_iterations:
        logging.info("Iteration %d: Processing error: %s", iteration + 1, error_message)
        result = agent.invoke({"input": error_message.strip()})
        response = result["response"]
        logging.info("Agent response:\n%s", response)
        rating = auto_evaluate_solution(response)
        logging.info("Auto-evaluated rating: %d", rating)
        if rating < 3:
            logging.info("Rating below threshold. Generating alternative solution.")
            alt_response = generate_alternative_solution(error_message)
            logging.info("Alternative response generated.")
            return alt_response
        else:
            return response
        iteration += 1
    logging.info("Max iterations reached. Returning last response.")
    return response

# Autonomous Agent Execution
def main():
    error_description = "Search results not displaying correctly"
    final_solution = get_solution_autonomously(error_description)
    print("\n=== Final Autonomous Response ===\n")
    print(final_solution)

    # subject = "Agent Report: RCA, Defect, Test Cases"
    # body = final_solution
    # sender_email = "rohumahajan0707@gmail.com"
    # s_password = "#H@num@nt713"
    # recipient_email = "rohannmahajan0707@gmail.com"

    # send_email(recipient_email, subject, body, sender_email, s_password)

if __name__ == "__main__":
    main()
