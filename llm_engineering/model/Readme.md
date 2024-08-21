# SageMaker Roles, Deployment, and Inference

This repository contains scripts for creating and managing AWS IAM roles and users for Amazon SageMaker, deploying a Hugging Face model as a SageMaker endpoint, and testing inference on the deployed endpoint.

## Contents

1. [AWS Configuration](#aws-configuration)
2. [SageMaker User Creation Script](#sagemaker-user-creation-script)
3. [SageMaker Execution Role Creation Script](#sagemaker-execution-role-creation-script)
4. [Understanding the Difference](#understanding-the-difference)
5. [Deploying a Hugging Face Inference Endpoint](#deploying-a-hugging-face-inference-endpoint)
6. [Testing Inference on the Deployed Endpoint](#testing-inference-on-the-deployed-endpoint)
7. [Using the Makefile](#using-the-makefile)

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
AWS Access Key ID [None]: <your access key ID>
AWS Secret Access Key [None]: <your secret access key>
Default region name [None]: us-west-2
Default output format [None]: json
```

4. This creates a default profile. If you want to create a named profile, use:
   ```
   aws configure --profile profilename
   ```

Now your AWS environment is set up and ready to use with the scripts in this repository.

## SageMaker User Creation Script

File: `llm_engineering/core/aws/create_sagemaker_role.py`

This script creates an IAM user with permissions to interact with SageMaker and other necessary AWS services.

### Features:
- Creates a new IAM user
- Attaches policies for full access to SageMaker, CloudFormation, IAM, ECR, and S3
- Generates and outputs access keys for programmatic access
- Saves the access keys to a JSON file

### Usage:
```
make create-sagemaker-role
```

## SageMaker Execution Role Creation Script

File: `llm_engineering/core/aws/create_sagemaker_execution_role.py`

This script creates an IAM role that SageMaker can assume to access other AWS resources on your behalf.

### Features:
- Creates a new IAM role with a trust relationship allowing SageMaker to assume the role
- Attaches policies for SageMaker, S3, CloudWatch Logs, and ECR access
- Outputs and saves the role ARN to a JSON file

### Usage:
```
make create-sagemaker-execution-role
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

File: `llm_engineering/model/deploy/huggingface/run.py`

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
```
make deploy-inference-endpoint
```

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

## Using the Makefile

This project includes a Makefile to automate common tasks and streamline the workflow. The Makefile provides several targets that correspond to the main operations in the project.

### Makefile Configuration

The Makefile sets the AWS profile to use:

```makefile
export AWS_PROFILE=decodingml
```

Ensure that you have this profile configured in your AWS CLI settings, or modify this line to match your desired AWS profile.

### Makefile Targets

- `help`: Displays a list of available commands with brief descriptions.
- `create-sagemaker-role`: Creates the SageMaker role.
- `create-sagemaker-execution-role`: Creates the SageMaker execution role.
- `deploy-inference-endpoint`: Deploys the inference endpoint.
- `delete-inference-endpoint`: Deletes the inference endpoint and its configuration.

### Usage

To use the Makefile, ensure you have `make` installed on your system. Then, you can run the following commands:

1. To see available commands:
   ```
   make help
   ```

2. To create a SageMaker role:
   ```
   make create-sagemaker-role
   ```

3. To create a SageMaker execution role:
   ```
   make create-sagemaker-execution-role
   ```

4. To deploy the inference endpoint:
   ```
   make deploy-inference-endpoint
   ```

5. To delete the inference endpoint:
   ```
   make delete-inference-endpoint ENDPOINT_NAME=<your-endpoint-name>
   ```
   Note: You must provide the ENDPOINT_NAME parameter when deleting an endpoint.

### Poetry Integration

This Makefile uses Poetry to manage Python dependencies and run scripts. Ensure you have Poetry installed and have run `poetry install` to set up your project environment.

### Note

- Ensure you have Python and Poetry installed on your system.
- The Makefile uses Poetry to run Python scripts, ensuring all dependencies are correctly managed.
- Make sure you have properly configured the AWS CLI and have the necessary permissions to perform these operations.
- When deleting an endpoint, you must provide the endpoint name as an environment variable.

By using this Makefile, you can easily manage the entire lifecycle of your SageMaker project, from setting up roles to deploying and managing your inference endpoints.

## Note
Ensure you have the necessary permissions in your AWS account to create IAM users and roles, deploy SageMaker endpoints, and perform inference before running these scripts.
