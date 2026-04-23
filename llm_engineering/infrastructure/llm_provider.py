from langchain_openai import ChatOpenAI
from loguru import logger

from llm_engineering.settings import settings

# Default context windows for supported providers
_PROVIDER_MAX_TOKEN_WINDOWS = {
    "openai": {
        "gpt-3.5-turbo": 16385,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
    },
    "minimax": {
        "MiniMax-M2.7": 204800,
        "MiniMax-M2.7-highspeed": 204800,
    },
}


def get_chat_model(temperature: float = 0, **kwargs) -> ChatOpenAI:
    """
    Create a ChatOpenAI-compatible LLM instance based on the configured provider.

    Supports 'openai' (default) and 'minimax' providers. MiniMax uses the
    OpenAI-compatible API, so ChatOpenAI works for both providers.

    Args:
        temperature: The sampling temperature. For MiniMax, 0 is clamped to 0.01
            since MiniMax requires temperature in (0.0, 1.0].
        **kwargs: Additional keyword arguments passed to ChatOpenAI.

    Returns:
        A ChatOpenAI instance configured for the selected provider.
    """
    provider = settings.LLM_PROVIDER.lower()

    if provider == "minimax":
        api_key = settings.MINIMAX_API_KEY
        if not api_key:
            raise ValueError(
                "MINIMAX_API_KEY must be set when using 'minimax' as LLM_PROVIDER. "
                "Get your API key at https://platform.minimax.io"
            )

        # MiniMax requires temperature in (0.0, 1.0]
        if temperature <= 0:
            temperature = 0.01

        model = ChatOpenAI(
            model=settings.MINIMAX_MODEL_ID,
            api_key=api_key,
            base_url="https://api.minimax.io/v1",
            temperature=temperature,
            **kwargs,
        )

        logger.debug(f"Using MiniMax provider with model '{settings.MINIMAX_MODEL_ID}'")
    else:
        # Default: OpenAI provider
        model = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
            **kwargs,
        )

        logger.debug(f"Using OpenAI provider with model '{settings.OPENAI_MODEL_ID}'")

    return model


def get_max_token_window() -> int:
    """
    Get the maximum token window for the currently configured provider and model.

    Returns:
        The maximum token window size (with a 10% safety margin applied).
    """
    provider = settings.LLM_PROVIDER.lower()

    if provider == "minimax":
        model_id = settings.MINIMAX_MODEL_ID
    else:
        model_id = settings.OPENAI_MODEL_ID

    provider_windows = _PROVIDER_MAX_TOKEN_WINDOWS.get(provider, {})
    official_max = provider_windows.get(model_id, 128000)

    return int(official_max * 0.90)
