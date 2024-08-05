import json

import boto3

from llm_engineering.settings import settings


def create_sagemaker_user(username, region_name="eu-central-1"):
    # Create IAM client
    iam = boto3.client(
        "iam",
        region_name=region_name,
        aws_access_key_id=settings.AWS_ACCESS_KEY,
        aws_secret_access_key=settings.AWS_SECRET_KEY,
    )

    # Create user
    iam.create_user(UserName=username)

    # Attach necessary policies
    policies = [
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/AWSCloudFormationFullAccess",
        "arn:aws:iam::aws:policy/IAMFullAccess",
        "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess",
    ]

    for policy in policies:
        iam.attach_user_policy(UserName=username, PolicyArn=policy)

    # Create access key
    response = iam.create_access_key(UserName=username)
    access_key = response["AccessKey"]

    print(f"User '{username}' created successfully.")
    print(f"Access Key ID: {access_key['AccessKeyId']}")
    print(f"Secret Access Key: {access_key['SecretAccessKey']}")

    # Return the access key info
    return {"AccessKeyId": access_key["AccessKeyId"], "SecretAccessKey": access_key["SecretAccessKey"]}


if __name__ == "__main__":
    new_user = create_sagemaker_user("sagemaker-deployer-2")

    # Save the access keys to a file
    with open("sagemaker_user_credentials.json", "w") as f:
        json.dump(new_user, f)

print("Credentials saved to 'sagemaker_user_credentials.json'")
