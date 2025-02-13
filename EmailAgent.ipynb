from google.colab import userdata
import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from typing import TypedDict, List

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Email credentials and addresses
email_password = "qcof scde ezte sxwn"
sender_email = "rohannmahajan0707@gmail.com"     # Sender's email
receiver_email = "rohumahajan0707@gmail.com"             # Receiver's email

# Helper function: Similarity computation
def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

# Import LangGraph and LangChain components
from langgraph.graph import END, StateGraph
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Initialize the LLM
groq_api = userdata.get("groq_api_key")
llm = ChatGroq(
    groq_api_key=groq_api,
    temperature=0.3,
    model_name="gemma2-9b-it",
)

# Load CSV files
try:
    defects_df = pd.read_csv("/content/defects.csv")
except Exception:
    defects_df = pd.DataFrame()

try:
    test_cases_df = pd.read_csv("/content/test_cases.csv")
except Exception:
    test_cases_df = pd.DataFrame(columns=["Module", "Test_Scenario", "Test_Steps", "Pre_Requisite", "Pass_Fail_Criteria", "Expected_Result"])

# Build documents from defects.csv (using columns: Description, Solution, Module)
docs = []
for _, row in defects_df.iterrows():
    if pd.notna(row.get("Description")) and pd.notna(row.get("Solution")):
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

# CSV Test Cases Fetching (without classification)
def get_csv_test_cases(module: str) -> List[dict]:
    test_cases = []
    global test_cases_df
    if test_cases_df.empty:
        return test_cases
    df_filtered = test_cases_df[test_cases_df["Module"] == module]
    for _, row in df_filtered.iterrows():
        tc_text = " ".join([
            str(row.get("Test_Scenario", "")),
            str(row.get("Test_Steps", "")),
            str(row.get("Pre_Requisite", "")),
            str(row.get("Expected_Result", "")),
            str(row.get("Pass_Fail_Criteria", ""))
        ])
        if tc_text.strip() == "":
            continue
        tc_dict = {
            "Module": row["Module"],
            "Test_Scenario": row["Test_Scenario"],
            "Test_Steps": row["Test_Steps"],
            "Pre_Requisite": row["Pre_Requisite"],
            "Pass_Fail_Criteria": row["Pass_Fail_Criteria"],
            "Expected_Result": row["Expected_Result"]
        }
        test_cases.append(tc_dict)
    return test_cases

# Helper to parse generated test cases into fields
def parse_test_case(tc_text: str) -> dict:
    fields = {
        "Test_Scenario": "",
        "Test_Steps": "",
        "Pre_Requisite": "",
        "Expected_Result": "",
        "Pass_Fail_Criteria": ""
    }
    for field in fields.keys():
        pattern = field + r":\s*(.*?)\s*(?=(Test_Steps:|Pre_Requisite:|Expected_Result:|Pass_Fail_Criteria:|$))"
        match = re.search(pattern, tc_text, re.DOTALL)
        if match:
            fields[field] = match.group(1).strip()
    return fields

# Helper to save new test cases to CSV (avoiding duplicates)
def save_new_test_cases(new_cases: List[dict]):
    global test_cases_df
    required_columns = ["Module", "Test_Scenario", "Test_Steps", "Pre_Requisite", "Pass_Fail_Criteria", "Expected_Result"]
    if test_cases_df.empty:
        test_cases_df = pd.DataFrame(columns=required_columns)
    rows_to_add = []
    for case in new_cases:
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

# Define Agent State
class AgentState(TypedDict):
    input: str
    context: List[Document]
    response: str

# Core Node: Validate or Generate Test Cases (with analysis of CSV test cases)
def validate_or_generate_test_cases(state: AgentState) -> dict:
    try:
        error_message = state["input"]
        solution = None
        module = None
        # Try to retrieve defect context from vector store.
        if state.get("context") and len(state["context"]) > 0:
            context = state["context"][0]
            if similarity(error_message.lower(), context.page_content.lower()) >= 0.6:
                solution = context.metadata["solution"]
                module = context.metadata["module"]
        # Fetch CSV test cases if module info is available.
        csv_test_cases = []
        if module:
            csv_test_cases = get_csv_test_cases(module)
       
        if csv_test_cases:
            # Format CSV test cases into plain text for analysis.
            def format_tc_plain(tc: dict) -> str:
                return (f"Test_Scenario: {tc['Test_Scenario']}\n"
                        f"Test_Steps: {tc['Test_Steps']}\n"
                        f"Pre_Requisite: {tc['Pre_Requisite']}\n"
                        f"Expected_Result: {tc['Expected_Result']}\n"
                        f"Pass_Fail_Criteria: {tc['Pass_Fail_Criteria']}")
            csv_tc_text = "\n\n".join(format_tc_plain(tc) for tc in csv_test_cases)
           
            # Ask LLM to analyze whether these test cases fully validate the solution.
            analysis_prompt = """
            [INST] Given the following solution:
            Solution: {solution}
            and the following test cases:
            {test_cases}
            Do these test cases fully validate the solution? If not, generate additional test cases that cover missing scenarios.
            Each additional test case MUST include:
              Test_Scenario: A short description of the scenario.
              Test_Steps: Step-by-step instructions.
              Pre_Requisite: Conditions before running the test.
              Expected_Result: What should happen.
              Pass_Fail_Criteria: How to determine if the test passes.
            End each additional test case with the delimiter "### END TEST CASE ###".
            [/INST]
            """
            # If solution was not found earlier, generate one.
            if not solution:
                gen_prompt = """
                [INST] Provide a clear solution for the following error and explain why it works:
                Error: {error}
                [/INST]
                """
                prompt_template = ChatPromptTemplate.from_template(gen_prompt)
                formatted_prompt = prompt_template.format_prompt(error=error_message)
                solution = llm.invoke(formatted_prompt.to_messages()).content.strip()
                explanation = "Solution generated as no defect was found in the database."
            else:
                explanation_prompt = """
                [INST] Explain why this solution fixes the following error:
                Error: {error}
                Solution: {solution}
                [/INST]
                """
                exp_template = ChatPromptTemplate.from_template(explanation_prompt)
                formatted_exp = exp_template.format_prompt(error=error_message, solution=solution)
                explanation = llm.invoke(formatted_exp.to_messages()).content.strip()
           
            # Analyze CSV test cases
            analysis_template = ChatPromptTemplate.from_template(analysis_prompt)
            formatted_analysis = analysis_template.format_prompt(solution=solution, test_cases=csv_tc_text)
            analysis_response = llm.invoke(formatted_analysis.to_messages()).content.strip()
            delimiter = "\n### END TEST CASE ###\n"
            additional_tc_raw = [tc.strip() for tc in re.split(delimiter, analysis_response) if tc.strip()]
            extra_generated_cases = []
            for tc_raw in additional_tc_raw:
                parsed = parse_test_case(tc_raw)
                if parsed["Test_Scenario"]:
                    extra_generated_cases.append({
                        "Module": module if module else "Generated",
                        "Test_Scenario": parsed["Test_Scenario"],
                        "Test_Steps": parsed["Test_Steps"],
                        "Pre_Requisite": parsed["Pre_Requisite"],
                        "Pass_Fail_Criteria": parsed["Pass_Fail_Criteria"],
                        "Expected_Result": parsed["Expected_Result"]
                    })
            # Combine CSV test cases and extra generated cases.
            all_test_cases = csv_test_cases + extra_generated_cases
           
            # Save the extra test cases to CSV (if any new ones were generated).
            if extra_generated_cases:
                save_new_test_cases(extra_generated_cases)
           
            # Format the combined test cases for HTML output.
            def format_tc(tc: dict) -> str:
                return (f"<p><strong>Test_Scenario:</strong> {tc['Test_Scenario']}<br>"
                        f"<strong>Test_Steps:</strong> {tc['Test_Steps']}<br>"
                        f"<strong>Pre_Requisite:</strong> {tc['Pre_Requisite']}<br>"
                        f"<strong>Expected_Result:</strong> {tc['Expected_Result']}<br>"
                        f"<strong>Pass_Fail_Criteria:</strong> {tc['Pass_Fail_Criteria']}</p>")
            test_cases_html = "\n".join(format_tc(tc) for tc in all_test_cases)
           
            response_template = (
                "<h2>Error:</h2><p>{Error}</p>"
                "<h2>Solution:</h2><p>{Solution}</p>"
                "<h2>Explanation:</h2><p>{Explanation}</p>"
                "<h2>Test Cases (from CSV + Generated):</h2>{TestCases}"
            )
            final_response = response_template.format(
                Error=error_message,
                Solution=solution,
                Explanation=explanation,
                TestCases=test_cases_html
            )
            state["response"] = final_response
            return {"response": final_response}
        else:
            # No CSV test cases available; generate comprehensive test cases end-to-end.
            if not solution:
                gen_prompt = """
                [INST] Provide a clear, actionable solution for the following error:
                Error: {error}
                [/INST]
                """
                prompt_template = ChatPromptTemplate.from_template(gen_prompt)
                formatted_prompt = prompt_template.format_prompt(error=error_message)
                solution = llm.invoke(formatted_prompt.to_messages()).content.strip()
                explanation = "Solution generated as no defect was found in the database."
            else:
                explanation_prompt = """
                [INST] Explain why this solution fixes the following error:
                Error: {error}
                Solution: {solution}
                [/INST]
                """
                exp_template = ChatPromptTemplate.from_template(explanation_prompt)
                formatted_exp = exp_template.format_prompt(error=error_message, solution=solution)
                explanation = llm.invoke(formatted_exp.to_messages()).content.strip()
            test_case_prompt = """
            [INST] Generate a comprehensive set of test cases to fully validate the following solution end-to-end.
            Error: {error}
            Solution: {solution}
            Explanation: {explanation}
            Your task is to cover all scenarios where the solution should work as well as those where it might fail.
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
            tc_template = ChatPromptTemplate.from_template(test_case_prompt)
            formatted_tc = tc_template.format_prompt(
                error=error_message, solution=solution, explanation=explanation)
            test_cases_response = llm.invoke(formatted_tc.to_messages()).content.strip()
            delimiter = "\n### END TEST CASE ###\n"
            test_cases_raw = [tc.strip() for tc in re.split(delimiter, test_cases_response) if tc.strip()]
            generated_cases = []
            for tc_raw in test_cases_raw:
                parsed = parse_test_case(tc_raw)
                if parsed["Test_Scenario"]:
                    generated_cases.append({
                        "Module": module if module else "Generated",
                        "Test_Scenario": parsed["Test_Scenario"],
                        "Test_Steps": parsed["Test_Steps"],
                        "Pre_Requisite": parsed["Pre_Requisite"],
                        "Pass_Fail_Criteria": parsed["Pass_Fail_Criteria"],
                        "Expected_Result": parsed["Expected_Result"]
                    })
            if generated_cases:
                save_new_test_cases(generated_cases)
            def format_tc(tc: dict) -> str:
                return (f"<p><strong>Test_Scenario:</strong> {tc['Test_Scenario']}<br>"
                        f"<strong>Test_Steps:</strong> {tc['Test_Steps']}<br>"
                        f"<strong>Pre_Requisite:</strong> {tc['Pre_Requisite']}<br>"
                        f"<strong>Expected_Result:</strong> {tc['Expected_Result']}<br>"
                        f"<strong>Pass_Fail_Criteria:</strong> {tc['Pass_Fail_Criteria']}</p>")
            test_cases_html = "\n".join(format_tc(tc) for tc in generated_cases)
            response_template = (
                "<h2>Error:</h2><p>{Error}</p>"
                "<h2>Solution:</h2><p>{Solution}</p>"
                "<h2>Explanation:</h2><p>{Explanation}</p>"
                "<h2>Test Cases (Generated):</h2>{TestCases}"
            )
            final_response = response_template.format(
                Error=error_message,
                Solution=solution,
                Explanation=explanation,
                TestCases=test_cases_html
            )
            state["response"] = final_response
            return {"response": final_response}
    except Exception as e:
        return {"response": f"Error processing request: {e}"}

# Function to Send Email with Formatted HTML Body
def send_email(final_solution: str):
    email_body = f"""<html>
  <head>
    <style>
      body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 10px; }}
      h2 {{ color: #2E86C1; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
      p {{ margin: 10px 0; }}
    </style>
  </head>
  <body>
    {final_solution}
  </body>
</html>"""
    message = MIMEMultipart('alternative', None, [MIMEText(email_body, 'html')])
    message['Subject'] = 'Defect RCA'
    message['From'] = sender_email
    message['To'] = receiver_email
    try:
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()
        server.login(sender_email, email_password)
        server.sendmail(sender_email, receiver_email, message.as_string())
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error in sending email: {e}")

# Build the StateGraph Workflow
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", lambda state: {"context": retriever.invoke(state["input"])})
workflow.add_node("validate_or_generate_test_cases", validate_or_generate_test_cases)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "validate_or_generate_test_cases")
workflow.add_edge("validate_or_generate_test_cases", END)
agent = workflow.compile()

# Automated Evaluation Functions (Optional)
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
    Generate a comprehensive set of test cases to validate the alternative solution end-to-end.
    Each test case MUST include:
      Test_Scenario
      Test_Steps
      Pre_Requisite
      Expected_Result
      Pass_Fail_Criteria
    End each test case with the delimiter "### END TEST CASE ###".
    [/INST]
    """
    tc_template = ChatPromptTemplate.from_template(test_case_prompt)
    formatted_tc = tc_template.format_prompt(error=error_message, solution=alternative_solution)
    alternative_test_cases = llm.invoke(formatted_tc.to_messages()).content.strip()

    alt_response = (
        "<h2>Alternative Solution (Generated):</h2><p>{AltSolution}</p>"
        "<h2>Test Cases for Alternative Solution:</h2><p>{AltTestCases}</p>"
    ).format(
        AltSolution=alternative_solution,
        AltTestCases=alternative_test_cases
    )
    return alt_response

def get_solution_autonomously(error_message: str) -> str:
    max_iterations = 3
    iteration = 0
    while iteration < max_iterations:
        result = agent.invoke({"input": error_message.strip()})
        response = result["response"]
        rating = auto_evaluate_solution(response)
        if rating < 3:
            alt_response = generate_alternative_solution(error_message)
            return alt_response
        else:
            return response
        iteration += 1
    return response

# Main Execution: Run Agent and Send Email
def main():
    error_description = "BIOS not booting up"
    final_solution = get_solution_autonomously(error_description)
    print("\n=== Final Autonomous Response ===\n")
    # print(final_solution)
    send_email(final_solution)

if __name__ == "__main__":
    main()
