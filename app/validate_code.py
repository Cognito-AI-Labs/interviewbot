import gradio as gr
import os
import re
from dotenv import load_dotenv
import logging
from sheets_util import get_google_sheets_service
from typing import Optional, Tuple, Any

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment and config
load_dotenv()
SCOPES: list[str] = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID: Optional[str] = os.getenv("SPREADSHEET_ID")
SHEET_NAME: str = "interview_tracker"
CURRENT_CODE: Optional[str] = None
CURRENT_EMAIL: Optional[str] = None
CURRENT_NAME: Optional[str] = None

def validate_email(email: str) -> bool:
    """
    Validates the format of an email address.

    Args:
        email (str): The email address to validate.

    Returns:
        bool: True if the email address is valid, False otherwise.
    """
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

def validate_code(
    code: str,
    name: str,
    email: str
) -> Tuple[gr.update, gr.update, gr.update, gr.update, bool]:
    """
    Validates a one-time code, candidate name, and email address against a Google Sheet.
    Marks the code as used and records the candidate's email and name if valid.

    Args:
        code (str): The one-time code provided by the candidate.
        name (str): The candidate's name.
        email (str): The candidate's email address.

    Returns:
        Tuple[gr.update, gr.update, gr.update, gr.update, bool]: 
            - Updates to hide the code, name, email, and submit button fields.
            - A boolean flag indicating successful validation.

    Raises:
        gr.Error: If validation fails due to empty name, invalid email, code not found, or code already used.
    """
    global CURRENT_CODE, CURRENT_EMAIL, CURRENT_NAME
    try:
        code = code.strip()
        name = name.strip()
        email = email.strip()

        logger.info(f"Validating code: {code}, name: {name}, email: {email}")

        if not name:
            logger.warning("Validation failed: Name is empty.")
            raise gr.Error("Name cannot be empty.")

        if not validate_email(email):
            logger.warning("Validation failed: Invalid email format.")
            raise gr.Error("Invalid email address.")

        service = get_google_sheets_service()
        sheet = service.spreadsheets()

        result: dict[str, Any] = sheet.values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{SHEET_NAME}!A:E"
        ).execute()
        rows: list[list[str]] = result.get("values", [])

        found_row: int = -1
        for i, row in enumerate(rows):
            if row and row[0] == code:
                found_row = i
                used: bool = row[2].strip().upper() == "TRUE" if len(row) > 2 else False
                if used:
                    logger.info(f"Code {code} already used at row {i+1}.")
                    raise gr.Error("This code has already been used.")
                break

        if found_row == -1:
            logger.warning(f"Code {code} not found.")
            raise gr.Error("Invalid code.")

        update_range: str = f"{SHEET_NAME}!C{found_row+1}:E{found_row+1}"
        body: dict[str, list[list[str]]] = {"values": [["TRUE", email, name]]}
        sheet.values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=update_range,
            valueInputOption="RAW",
            body=body
        ).execute()

        logger.info(f"Code {code} validated and marked as used for {email}.")

        CURRENT_CODE = code
        CURRENT_EMAIL = email
        CURRENT_NAME = name

        return (
            gr.update(visible=False),  # one_time_id
            gr.update(visible=False),  # name_input
            gr.update(visible=False),  # email_input
            gr.update(visible=False),  # submit button
            True,                      # success flag
        )

    except gr.Error as ge:
        logger.warning(f"Validation error: {ge}")
        raise

    except Exception as e:
        logger.exception("Unexpected error during code validation.")
        raise gr.Error("Internal error occurred. Please try again later.")
