from llm_engineering.infrastructure.inference_pipeline_api import app  # noqa

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("tools.ml_service:app", host="0.0.0.0", port=8000, reload=True)
