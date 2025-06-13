#!/usr/bin/env python3
import os

import aws_cdk as cdk

from interviewbot_cdk.interviewbot_cdk_stack import InterviewBotStack
from aws_cdk import Environment



app = cdk.App()

InterviewBotStack(app, "InterviewBotStack",
    env=Environment(
        account=os.getenv('CDK_DEFAULT_ACCOUNT'),
        region=os.getenv('CDK_DEFAULT_REGION', 'us-east-1')
    )
)


app.synth()
