import boto3
import gradio as gr
import os
import re
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
TABLE_NAME = os.getenv("DYNAMODB_TABLE", "InterviewCodes")

session = boto3.Session()
dynamodb = session.resource("dynamodb")
# dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(TABLE_NAME)
CURRENT_CODE = None
CURRENT_EMAIL = None
def validate_email(email: str) -> bool:
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

def validate_code(code: str, email: str):
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
    return (
        gr.update(visible=False),  
        gr.update(visible=False), 
        gr.update(visible=False),  
        gr.update(visible=True)   
    )


