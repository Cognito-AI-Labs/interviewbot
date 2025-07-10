import streamlit as st
import random
import os
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID = os.environ["SPREADSHEET_ID"]
SHEET_NAME = 'interview_tracker'
RANGE_TO_READ = f"{SHEET_NAME}!A:A"

def get_sheets_service():
    SERVICE_ACCOUNT_FILE = os.path.join(os.getcwd(), "credentials.json")
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return build("sheets", "v4", credentials=credentials)

def generate_code():
    return ''.join([str(random.randint(0, 9)) for _ in range(12)])

def code_exists(service, code):
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_TO_READ).execute()
    values = result.get("values", [])
    existing_codes = {row[0] for row in values if row}
    return code in existing_codes

def append_code(service, code, prompt):
    body = {"values": [[code, prompt, "FALSE"]]}
    sheet = service.spreadsheets()
    sheet.values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{SHEET_NAME}!A:C",
        valueInputOption="RAW",
        body=body
    ).execute()

st.title("🎯 Interview Code Generator")

prompt = st.text_area("Enter the prompt for interview:")

if st.button("Generate New Interview Code"):
    if not prompt.strip():
        st.warning("Please enter the prompt before generating a code.")
    else:
        service = get_sheets_service()
        for _ in range(5):
            code = generate_code()
            if not code_exists(service, code):
                append_code(service, code, prompt)
                st.success(f"✅ New Code Generated: `{code}`")
                break
        else:
            st.error("⚠️ Could not generate a unique code after 5 attempts.")
