import streamlit as st
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import os, json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from google import genai
from pydantic import BaseModel
from typing import List

load_dotenv()
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
SHEET_NAME = "interview_tracker"
TRANSCRIPT_COL = 8
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_google_sheets_service():
    SERVICE_ACCOUNT_FILE = os.path.join(os.getcwd(), "credentials.json")
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return build("sheets", "v4", credentials=credentials)

@st.cache_data
def get_sheet_data():
    try:
        sheet = get_google_sheets_service().spreadsheets()
        result = sheet.values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{SHEET_NAME}!A:Z"
        ).execute()

        rows = result.get("values", [])
        if not rows:
            return pd.DataFrame()
        headers = rows[0]
        data = rows[1:]
        # Ensure all data rows have the same number of columns as headers
        normalized_data = [
            row + [""] * (len(headers) - len(row)) if len(row) < len(headers) else row
            for row in data
        ]
        return pd.DataFrame(normalized_data, columns=headers)

    except Exception as e:
        st.error(f"Error loading data from Google Sheets: {e}")
        return pd.DataFrame()

def fetch_transcript_by_code(code):
    df = get_sheet_data()
    match = df[df['code'] == code]
    if not match.empty and len(match.columns) > TRANSCRIPT_COL:
        return match.iloc[0, TRANSCRIPT_COL]
    return None
# Gemini evaluation schema
class QuestionAssessment(BaseModel):
    question: str
    candidate_response: str
    answer_correctness: str
    feedback: str

class TopicEvaluation(BaseModel):
    topic: str
    assessments: List[QuestionAssessment]

class InterviewEvaluation(BaseModel):
    topics: List[TopicEvaluation]

# Evaluation using Gemini
def evaluate_transcript(transcript):
    prompt = f"""
    # Technical Interview Evaluation Task
    ## Interview Transcript
    {transcript}
    ## Evaluation Instructions:
    Analyze this technical interview transcript and provide a structured evaluation of the candidate's performance.
    For each question, state if it was answered correctly (Yes/No), assess technical understanding, and give feedback.
    Return output in JSON structured by topics.
    """
    client = genai.Client(api_key=GOOGLE_API_KEY)
    result = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': InterviewEvaluation,
        }
    )
    return json.loads(result.text)

# Streamlit UI
st.set_page_config(page_title="Candidate Reports", layout="wide")
st.title("📋 Candidate Interview Reports")
df = get_sheet_data()
if "start_time" not in df.columns:
    st.error("The column 'start_time' is missing from the sheet.")
    st.stop()
# Extract and show unique dates
df["date"] = pd.to_datetime(df["start_time"], errors="coerce").dt.date
unique_dates = sorted(df["date"].dropna().unique())
selected_date = st.selectbox("📅 Select a date:", unique_dates)
filtered_df = df[df["date"] == selected_date]
if not filtered_df.empty:
    code_options = filtered_df[["code", "candidate_name"]].dropna()
    if code_options.empty:
        st.warning("No candidates found for selected date.")
    else:
        selected_row = st.selectbox("👤 Select a candidate (code - name):",
                            code_options.apply(lambda row: f"{row['code']} - {row['candidate_name']}", axis=1))
        if selected_row:
            code = selected_row.split(" - ")[0]
            if st.button("📊 Show Evaluation Report"):
                transcript = fetch_transcript_by_code(code)
                if not transcript:
                    st.error("Transcript not found.")
                else:
                    with st.spinner("Evaluating transcript..."):
                        try:
                            eval_json = evaluate_transcript(transcript)
                            rows = []
                            for topic in eval_json['topics']:
                                for a in topic['assessments']:
                                    rows.append({
                                        "Topic": topic['topic'],
                                        "Question": a['question'],
                                        "Candidate Response": a['candidate_response'],
                                        "Correct": a['answer_correctness'],
                                        "Feedback": a['feedback']
                                    }) 
                            # Convert to DataFrame
                            df_eval = pd.DataFrame(rows)
                            # Display table
                            st.subheader("📊 Evaluation Report")
                            st.dataframe(df_eval, use_container_width=True)

                        except Exception as e:
                            st.error(f"Failed to evaluate: {e}")
else:
    st.info("No interviews recorded for the selected date.")
