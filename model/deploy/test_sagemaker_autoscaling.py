import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import boto3
from ai_document_tasks.core.aws.utils import authenticate_with_aws_vault
from ai_document_tasks.settings import settings

authenticate_with_aws_vault("epostbox.development")

inference_component_name = "huggingface-pytorch-tgi-inference-2024-02-12-16-1707754042-55ea"
endpoint_name = settings.SAGEMAKER_ENDPOINT_SUMMARIZATION
sm_client = boto3.client("sagemaker")
smr_client = boto3.client("sagemaker-runtime")
max_copy_count_per_instance = 10
max_instance_count = 10
initial_copy_count = 1


@dataclass
class AutoscalingStatus:
    status_name: str  # endpoint status or inference component status
    start_time: datetime  # when was the status changed
    current_instance_count: int
    desired_instance_count: int
    current_copy_count: int
    desired_copy_count: int


class WorkerThread(threading.Thread):
    def __init__(self, do_run, *args, **kwargs):
        super(WorkerThread, self).__init__(*args, **kwargs)
        self.__do_run = do_run
        self.__terminate_event = threading.Event()

    def terminate(self):
        self.__terminate_event.set()

    def is_terminated(self):
        return self.__terminate_event.is_set()

    def run(self):
        while not self.__terminate_event.is_set():
            self.__do_run(self.__terminate_event)


invoke_endpoint_sanity_check_sample = {
    "inputs": "The diamondback terrapin was the first reptile to be",
    "parameters": {
        "do_sample": True,
        "max_new_tokens": 100,
        "min_new_tokens": 100,
        "temperature": 0.3,
        "watermark": True,
    },
}
invoke_endpoint_sanity_check_payload = json.dumps(invoke_endpoint_sanity_check_sample)


def invoke_endpoint_sanity_check(
    sagemaker_runtime_client, endpoint_name, container_names=None, inference_component_name=None
):
    try:
        parameters = {
            "EndpointName": endpoint_name,
            "ContentType": "application/json",
            "Body": invoke_endpoint_sanity_check_payload,
        }
        if container_names is not None:
            for container_name in container_names:
                parameters["TargetContainerHostname"] = container_name
                response = sagemaker_runtime_client.invoke_endpoint(**parameters)
                print("Inference Result:", response["Body"].read().decode("utf-8"))
        else:
            if inference_component_name is not None:
                parameters["InferenceComponentName"] = inference_component_name
            response = sagemaker_runtime_client.invoke_endpoint(**parameters)
            print("Inference Result:", response["Body"].read().decode("utf-8"))
    except Exception as e:
        print(f"Failed to invoke {endpoint_name}: " + str(e))


def invoke_endpoint(terminate_event):
    start_time = datetime.utcnow()
    for _ in range(max_copy_count_per_instance * max_instance_count * 2):
        invoke_endpoint_sanity_check(smr_client, endpoint_name, inference_component_name=inference_component_name)
        time.sleep(0.1)
    elapsed_seconds = (datetime.utcnow() - start_time).total_seconds()
    if terminate_event.is_set():
        return
    if elapsed_seconds < 60:
        time.sleep(60 - elapsed_seconds)


# Keep invoking the endpoint with test data
invoke_endpoint_thread = WorkerThread(do_run=invoke_endpoint)
invoke_endpoint_thread.start()

statuses = []
while True:
    endpoint_desc = sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = endpoint_desc["EndpointStatus"]
    current_instance_count = endpoint_desc["ProductionVariants"][0]["CurrentInstanceCount"]
    desired_instance_count = endpoint_desc["ProductionVariants"][0]["DesiredInstanceCount"]
    ic_desc = sm_client.describe_inference_component(InferenceComponentName=inference_component_name)
    ic_status = ic_desc["InferenceComponentStatus"]
    current_copy_count = ic_desc["RuntimeConfig"]["CurrentCopyCount"]
    desired_copy_count = ic_desc["RuntimeConfig"]["DesiredCopyCount"]
    status_name = f"{status}_{ic_status}"
    if not statuses or statuses[-1].status_name != status_name:
        statuses.append(
            AutoscalingStatus(
                status_name=status_name,
                start_time=datetime.utcnow(),
                current_instance_count=current_instance_count,
                desired_instance_count=desired_instance_count,
                current_copy_count=current_copy_count,
                desired_copy_count=desired_copy_count,
            )
        )
        print(statuses[-1])
    if status_name == "InService_InService":
        if current_copy_count == 20:
            invoke_endpoint_thread.terminate()
        elif current_copy_count == initial_copy_count:
            if invoke_endpoint_thread.is_terminated():
                break
    time.sleep(1)
