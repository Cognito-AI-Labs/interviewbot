import boto3
import gradio as gr
import os
import re
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
TABLE_NAME = os.getenv("DYNAMODB_TABLE", "InterviewCodes")

# Use SSO-based profile if needed
# session = boto3.Session(profile_name="interviewbot")
# dynamodb = session.resource("dynamodb", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(TABLE_NAME)

# Shared global variables for later use (e.g., file naming)
CURRENT_CODE = None
CURRENT_EMAIL = None

def validate_email(email: str) -> bool:
    """Basic email format check"""
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

def validate_code(code: str, email: str):
    """Validate code and email, mark used, and store email in DynamoDB"""
    global CURRENT_CODE, CURRENT_EMAIL

    code = code.strip()
    email = email.strip()

    if not validate_email(email):
        raise gr.Error("Invalid email address.")

    response = table.get_item(Key={"code": code})
    item = response.get("Item")

    if not item:
        raise gr.Error("Invalid code.")
    if item.get("used", False):
        raise gr.Error("This code has already been used.")

    # Mark code as used and save email
    table.update_item(
        Key={"code": code},
        UpdateExpression="SET used = :val1, email = :val2",
        ExpressionAttributeValues={
            ":val1": True,
            ":val2": email
        }
    )

    CURRENT_CODE = code
    CURRENT_EMAIL = email

    # Hide inputs and show interview UI
    return (
        gr.update(visible=False),  # Hide code input
        gr.update(visible=False),  # Hide email input
        gr.update(visible=False),  # Hide submit button
        gr.update(visible=True)    # Show interview UI
    )
