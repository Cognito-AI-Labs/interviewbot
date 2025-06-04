import streamlit as st
import boto3
import random
import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
TABLE_NAME = os.getenv("DYNAMODB_TABLE")

session = boto3.Session(profile_name="interviewbot")
dynamodb = session.resource("dynamodb", region_name="us-east-1")
table = dynamodb.Table(TABLE_NAME)

def generate_code():
    return ''.join([str(random.randint(0, 9)) for _ in range(12)])

def code_exists(code):
    response = table.get_item(Key={"code": code})
    return "Item" in response

def save_code(code):
    table.put_item(Item={"code": code, "used": False})

st.title("🎯 Interview Code Generator")

if st.button("Generate Code"):
    for _ in range(5):  
        code = generate_code()
        if not code_exists(code):
            save_code(code)
            st.success(f"✅ New code: {code}")
            break
    else:
        st.error("⚠️ Failed to generate a unique code. Try again.")
