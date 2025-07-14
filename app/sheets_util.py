# sheets_util.py
import os
import pathlib
from typing import Optional, Any
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import Resource, build
import validate_code
from dotenv import load_dotenv

load_dotenv()
current_dir = pathlib.Path(__file__).parent
SPREADSHEET_ID: str = os.environ["SPREADSHEET_ID"]

def get_google_sheets_service() -> Resource:
    """
    Initializes and returns a Google Sheets API service client using service account credentials.

    Returns:
        Resource: An authorized Google Sheets API service client.
    """
    SERVICE_ACCOUNT_FILE = current_dir / "credentials.json"
    credentials: Credentials = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return build("sheets", "v4", credentials=credentials)

def fetch_prompt_from_sheet() -> Optional[str]:
    """
    Fetches the interview prompt from the Google Sheet for the current code.

    Returns:
        Optional[str]: The prompt string if found, otherwise None.
    """
    try:
        service: Resource = get_google_sheets_service()
        sheet = service.spreadsheets()
        result: dict[str, Any] = sheet.values().get(
            spreadsheetId=SPREADSHEET_ID,
            range="interview_tracker!A2:B"
        ).execute()
        values: list[list[str]] = result.get("values", [])
        for row in values:
            if len(row) >= 2 and row[0] == validate_code.CURRENT_CODE:
                return row[1]
    except Exception as e:
        import logging
        logging.error(f"Failed to fetch prompt: {e}")
        return None
