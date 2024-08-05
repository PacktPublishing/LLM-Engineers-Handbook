# SageMaker Roles, Deployment, and Inference

This repository contains scripts for creating and managing AWS IAM roles and users for Amazon SageMaker, deploying a Hugging Face model as a SageMaker endpoint, and testing inference on the deployed endpoint.

## Contents

1. [AWS Configuration](#aws-configuration)
2. [SageMaker User Creation Script](#sagemaker-user-creation-script)
3. [SageMaker Execution Role Creation Script](#sagemaker-execution-role-creation-script)
4. [Understanding the Difference](#understanding-the-difference)
5. [Deploying a Hugging Face Inference Endpoint](#deploying-a-hugging-face-inference-endpoint)
6. [Testing Inference on the Deployed Endpoint](#testing-inference-on-the-deployed-endpoint)

## AWS Configuration

Before you can use the scripts in this repository, you need to set up your AWS environment. This involves creating an IAM user, installing the AWS CLI, and configuring your AWS profile.

### Creating an IAM User

1. Sign in to the AWS Management Console and open the IAM console at https://console.aws.amazon.com/iam/
2. In the navigation pane, choose Users and then choose Add user.
3. Type the user name for the new user.
4. Select Programmatic access as the AWS access type.
5. Choose Next: Permissions.
6. Set permissions for the user. For this project, you may want to attach the AmazonSageMakerFullAccess policy. However, for production environments, it's recommended to create a custom policy with only the necessary permissions.
7. Choose Next: Tags (optional to add tags).
8. Choose Next: Review to see all of the choices you made up to this point.
9. Choose Create user.
10. Download or copy the access key ID and secret access key. You will need these to configure the AWS CLI.

### Installing the AWS CLI

1. Follow the official AWS documentation to install the AWS CLI for your operating system:
   https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html

### Configuring Your AWS Profile

1. Open a terminal or command prompt.
2. Run the following command:
   ```
   aws configure
   ```
3. You will be prompted to enter your AWS Access Key ID, AWS Secret Access Key, default region name, and default output format. Enter the information you obtained when creating the IAM user.

Example:
```
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
Default region name [None]: us-west-2
Default output format [None]: json
```

4. This creates a default profile. If you want to create a named profile, use:
   ```
   aws configure --profile profilename
   ```

Now your AWS environment is set up and ready to use with the scripts in this repository.

## SageMaker User Creation Script

File: `create_sagemaker_user.py`

This script creates an IAM user with permissions to interact with SageMaker and other necessary AWS services.

### Features:
- Creates a new IAM user
- Attaches policies for full access to SageMaker, CloudFormation, IAM, ECR, and S3
- Generates and outputs access keys for programmatic access
- Saves the access keys to a JSON file

### Usage:
```python
python create_sagemaker_user.py
```

## SageMaker Execution Role Creation Script

File: `create_sagemaker_execution_role.py`

This script creates an IAM role that SageMaker can assume to access other AWS resources on your behalf.

### Features:
- Creates a new IAM role with a trust relationship allowing SageMaker to assume the role
- Attaches policies for SageMaker, S3, CloudWatch Logs, and ECR access
- Outputs and saves the role ARN to a JSON file

### Usage:
```python
python create_sagemaker_execution_role.py
```

## Understanding the Difference

### SageMaker User Role
- Purpose: For human users or applications to access AWS services
- Authentication: Uses access keys for authentication
- Usage: Used in scripts or applications that manage SageMaker resources

### SageMaker Execution Role
- Purpose: For SageMaker to access other AWS resources on your behalf
- Authentication: Uses temporary credentials via AssumeRole
- Usage: Provided to SageMaker when creating notebooks, training jobs, or deploying models

### Key Differences
1. **Purpose**: User roles are for external access to AWS. Execution roles are for internal AWS service-to-service access.
2. **Authentication**: User roles use long-term access keys. Execution roles use short-term credentials.
3. **Trust Relationship**: Execution roles have a trust relationship with the SageMaker service.
4. **Usage Context**: User roles are used in your code to interact with AWS. Execution roles are used by SageMaker itself.

## Deploying a Hugging Face Inference Endpoint

After setting up the necessary AWS resources (user and execution role), you can deploy a Hugging Face model as a SageMaker inference endpoint.

File: `deploy_huggingface_endpoint.py`

This script creates a SageMaker endpoint for inference using a Hugging Face model.

### Features:
- Uses the Hugging Face LLM image for SageMaker
- Configures the endpoint based on settings in `llm_engineering.settings`
- Supports different endpoint types (MODEL_BASED or INFERENCE_COMPONENT_BASED)
- Uses `SagemakerHuggingfaceStrategy` for deployment

### Prerequisites:
- Ensure you have set up the SageMaker execution role and user as described in the previous sections
- Configure your settings in `llm_engineering.settings`, including:
  - `GPU_INSTANCE_TYPE`
  - `SAGEMAKER_ENDPOINT_INFERENCE`
  - `SAGEMAKER_ENDPOINT_CONFIG_INFERENCE`
  - `ARN_ROLE` (the ARN of your SageMaker execution role)

### Usage:
```python
python deploy_huggingface_endpoint.py
```

### Code Overview:
```python
from sagemaker.enums import EndpointType
from sagemaker.huggingface import get_huggingface_llm_image_uri
from llm_engineering.model.deploy.huggingface.config import (
    hugging_face_deploy_config,
    model_resource_config,
)
from llm_engineering.model.deploy.huggingface.sagemaker_huggingface import (
    DeploymentService,
    SagemakerHuggingfaceStrategy,
)
from llm_engineering.model.utils import ResourceManager
from llm_engineering.settings import settings

def create_huggingface_endpoint(task_name, endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED):
    # ... [function implementation] ...

if __name__ == "__main__":
    task_name = "inference"
    create_huggingface_endpoint(task_name, endpoint_type=EndpointType.MODEL_BASED)
```

This script sets up and deploys a Hugging Face model as a SageMaker endpoint. It uses the SageMaker execution role you created earlier to give the endpoint the necessary permissions to access AWS resources.

### Note:
- Ensure all required dependencies are installed and AWS credentials are properly configured.
- The deployment process may take some time depending on the size of the model and the selected instance type.
- Monitor the AWS console or CloudWatch logs for deployment progress and any potential issues.

## Testing Inference on the Deployed Endpoint

After successfully deploying the Hugging Face model as a SageMaker endpoint, you can test the inference capabilities using the provided script.

File: `test.py`

This script demonstrates how to use the deployed endpoint for inference tasks.

### Features:
- Connects to the deployed SageMaker endpoint
- Allows customization of input text and prompt
- Supports parameter tuning (max_new_tokens, repetition_penalty, temperature)
- Returns the generated text based on the input

### Prerequisites:
- Ensure the SageMaker endpoint is successfully deployed
- Configure your settings in `llm_engineering.settings`, including:
  - `SAGEMAKER_ENDPOINT_INFERENCE`
  - `MAX_NEW_TOKENS_INFERENCE`
  - `TEMPERATURE_INFERENCE`

### Usage:
```python
python test.py
```

### Code Overview:
```python
from __future__ import annotations
import logging
from llm_engineering.core.interfaces import Inference
from llm_engineering.model.inference.inference import LLMInferenceSagemakerEndpoint
from llm_engineering.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class InferenceExecutor:
    def __init__(self, llm: Inference, text: str, prompt: str):
        self.llm = llm
        self.text = text
        self.prompt = prompt

    def execute(self) -> str:
        """Extracts entities from a text."""
        self.llm.set_payload(
            inputs=self.prompt.format(TEXT=self.text),
            parameters={
                "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,
                "repetition_penalty": 1.1,
                "temperature": settings.TEMPERATURE_INFERENCE,
            },
        )
        extraction = self.llm.inference()[0]["generated_text"]
        return extraction

if __name__ == "__main__":
    text = "The weather in Berlin is nice today."
    prompt = 'Continue the following text: "{TEXT}"'
    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
    )
    result = InferenceExecutor(llm, text, prompt).execute()
    print(f"Generated text: {result}")
```

This script sets up an `InferenceExecutor` class that handles the interaction with the SageMaker endpoint. It takes an input text and a prompt, sends them to the endpoint, and returns the generated text.

### Customization:
- You can modify the `text` and `prompt` variables in the `__main__` section to test different inputs.
- Adjust the parameters in `set_payload()` method to fine-tune the inference process.

### Note:
- Ensure all required dependencies are installed and AWS credentials are properly configured.
- The inference process may take a few seconds depending on the model size and input complexity.
- Monitor the console output for the generated text and any potential errors.

## Note
Ensure you have the necessary permissions in your AWS account to create IAM users and roles, deploy SageMaker endpoints, and perform inference before running these scripts.
