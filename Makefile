export AWS_PROFILE=decodingml

.PHONY: help create-sagemaker-role create-sagemaker-execution-role deploy-inference-endpoint delete-inference-endpoint

help:
	@echo "Available commands:"
	@echo " create-sagemaker-role            Create the SageMaker role."
	@echo " create-sagemaker-execution-role  Create the SageMaker execution role."
	@echo " deploy-inference-endpoint        Deploy the inference endpoint."
	@echo " delete-inference-endpoint        Delete the inference endpoint and config."

create-sagemaker-role:
	@echo "Creating the SageMaker role..."
	poetry run python llm_engineering/core/aws/roles/create_sagemaker_role.py

create-sagemaker-execution-role:
	@echo "Creating the SageMaker execution role..."
	poetry run python llm_engineering/core/aws/roles/create_execution_role.py

deploy-inference-endpoint:
	@echo "Deploying the inference endpoint..."
	poetry run python llm_engineering/model/deploy/huggingface/run.py

delete-inference-endpoint:
	@if [ -z "$(ENDPOINT_NAME)" ]; then \
		echo "Error: ENDPOINT_NAME is not set. Usage: make delete-inference-endpoint ENDPOINT_NAME=<name>"; \
		exit 1; \
	fi
	@echo "Deleting the inference endpoint and config..."
	poetry run python llm_engineering/model/delete_inference_endpoint.py $(ENDPOINT_NAME)

test-inference:
	poetry run python -m llm_engineering.model.inference.test