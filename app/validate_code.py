import gradio as gr
import os
import re
from dotenv import load_dotenv
import pathlib
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2 import service_account
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
load_dotenv()
# Google Sheets setup
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID = os.environ["SPREADSHEET_ID"]
SHEET_NAME = "interview_tracker"
CURRENT_CODE = None
CURRENT_EMAIL = None
current_dir = pathlib.Path(__file__).parent

def get_google_sheets_service():
    try:
        SERVICE_ACCOUNT_FILE = current_dir / "credentials.json"
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        return build("sheets", "v4", credentials=credentials)
    except Exception as e:
        logger.error(f'Failed to fetch the sheets service {e}.')

def validate_email(email: str) -> bool:
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

def validate_code(code: str, name: str, email: str):
    global CURRENT_CODE, CURRENT_EMAIL, CURRENT_NAME
    code = code.strip()
    name = name.strip()
    email = email.strip()
    if not name:
        raise gr.Error("Name cannot be empty.")
    if not validate_email(email):
        raise gr.Error("Invalid email address.")
    service = get_google_sheets_service()
    sheet = service.spreadsheets()
    result = sheet.values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{SHEET_NAME}!A:E"  
    ).execute()
    rows = result.get("values", [])
    found_row = -1

    for i, row in enumerate(rows):
        if row and row[0] == code:
            found_row = i
            used = row[2].strip().upper() == "TRUE" if len(row) > 1 else False
            if used:
                raise gr.Error("This code has already been used.")
            break

    if found_row == -1:
        raise gr.Error("Invalid code.")

    update_range = f"{SHEET_NAME}!C{found_row+1}:E{found_row+1}"
    body = {"values": [["TRUE", email, name]]}
    sheet.values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=update_range,
        valueInputOption="RAW",
        body=body
    ).execute()

    CURRENT_CODE = code
    CURRENT_EMAIL = email
    CURRENT_NAME = name

    return (
        gr.update(visible=False),  # one_time_id
        gr.update(visible=False),  # name_input
        gr.update(visible=False),  # email_input
        gr.update(visible=False), # submit button
        gr.update(visible=True),   # interview_ui
    )
