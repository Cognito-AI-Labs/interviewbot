from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_logs as logs,
    aws_elasticloadbalancingv2 as elbv2,
    aws_iam as iam
)
import os
from dotenv import load_dotenv
load_dotenv()
from constructs import Construct
from aws_cdk import CfnOutput


class InterviewBotStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # VPC and ECS Cluster
        vpc = ec2.Vpc(self, "InterviewBotVPC", max_azs=2)
        cluster = ecs.Cluster(self, "InterviewBotCluster", vpc=vpc)

        # IAM Role for Task
        task_role = iam.Role(self, "InterviewBotTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonDynamoDBFullAccess")
            ]
        )

        # Use image already pushed to ECR
        account_id = Stack.of(self).account
        region = Stack.of(self).region
        ecr_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/interviewbot-app:latest"
        execution_role = iam.Role(self, "InterviewBotExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonECSTaskExecutionRolePolicy"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryReadOnly")
            ]
        )
        # Task Definition
        task_definition = ecs.FargateTaskDefinition(self, "InterviewBotTaskDef",
            cpu=512,
            memory_limit_mib=1024,
            task_role=task_role,
            execution_role=execution_role
        )

        container = task_definition.add_container("InterviewBotContainer",
            image=ecs.ContainerImage.from_registry(ecr_image_uri),
            logging=ecs.LogDrivers.aws_logs(stream_prefix="InterviewBot"),
            environment={
                "GOOGLE_API_KEY": os.environ["GOOGLE_API_KEY"],
                "AWS_REGION": os.getenv("AWS_REGION", "us-east-1"),
                "S3_BUCKET": os.environ["S3_BUCKET"],
                "DYNAMODB_TABLE": os.environ["DYNAMODB_TABLE"]
            }
        )
        container.add_port_mappings(ecs.PortMapping(container_port=7860))

        # ECS Fargate Service
        service = ecs.FargateService(self, "InterviewBotService",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=1
        )

        # Load Balancer
        lb = elbv2.ApplicationLoadBalancer(self, "InterviewBotLB", vpc=vpc, internet_facing=True)
        listener = lb.add_listener("Listener", port=80)
        listener.add_targets("Target",
        port=7860,
        protocol=elbv2.ApplicationProtocol.HTTP,
        targets=[service]
    )


        self.load_balancer_dns = lb.load_balancer_dns_name
        CfnOutput(self, "AppURL",
        value=f"http://{lb.load_balancer_dns_name}:7860",
        description="URL to access the Interview Bot app"
    )