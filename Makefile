.PHONY: deploy-inference-endpoint


create-sagemaker-role:
	poetry run python llm_engineering/core/aws/create_sagemaker_role.py

create-sagemaker-execution-role:
	poetry run python llm_engineering/core/aws/create_sagemaker_execution_role.py
	
deploy-inference-endpoint:
	poetry run python llm_engineering/model/deploy/huggingface/run.py

