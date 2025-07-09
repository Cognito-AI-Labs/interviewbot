import streamlit as st
import random
import os
from dotenv import load_dotenv

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()

# ------------- Google Sheets Config -------------
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID = os.environ["SPREADSHEET_ID"]
SHEET_NAME = 'interview_tracker'
RANGE_TO_READ = f"{SHEET_NAME}!A:A"

# ------------- Google Sheets Service -------------
def get_sheets_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("sheets", "v4", credentials=creds)

# ------------- Code Logic -------------
def generate_code():
    return ''.join([str(random.randint(0, 9)) for _ in range(12)])

def code_exists(service, code):
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_TO_READ).execute()
    values = result.get("values", [])
    existing_codes = {row[0] for row in values if row}
    return code in existing_codes

def append_code(service, code):
    body = {"values": [[code, "", "FALSE"]]}
    sheet = service.spreadsheets()
    sheet.values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{SHEET_NAME}!A:C",
        valueInputOption="RAW",
        body=body
    ).execute()

# ------------- Streamlit App -------------
st.title("🎯 Interview Code Generator")

if st.button("Generate New Interview Code"):
    service = get_sheets_service()
    
    for _ in range(5):
        code = generate_code()
        if not code_exists(service, code):
            append_code(service, code)
            st.success(f"✅ New Code Generated: `{code}`")
            break
    else:
        st.error("⚠️ Could not generate a unique code after 5 attempts.")
