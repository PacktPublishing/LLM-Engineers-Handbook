"""Integration tests for MiniMax LLM provider.

These tests make real API calls to MiniMax and require MINIMAX_API_KEY to be set.
Skip with: pytest -m "not integration"
"""

import importlib
import importlib.util
import os
import sys
from types import ModuleType
from unittest.mock import patch

import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict

pytestmark = pytest.mark.integration


class _IntegrationSettings(BaseSettings):
    """Settings class for integration tests."""

    model_config = SettingsConfigDict(env_file=None)

    LLM_PROVIDER: str = "minimax"
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_API_KEY: str | None = None
    MINIMAX_API_KEY: str | None = None
    MINIMAX_MODEL_ID: str = "MiniMax-M2.7"


# Set up fake package hierarchy to load llm_provider in isolation
if "llm_engineering" not in sys.modules:
    _pkg = ModuleType("llm_engineering")
    _pkg.__path__ = []
    sys.modules["llm_engineering"] = _pkg

if "llm_engineering.infrastructure" not in sys.modules:
    _infra = ModuleType("llm_engineering.infrastructure")
    _infra.__path__ = []
    sys.modules["llm_engineering.infrastructure"] = _infra

if "llm_engineering.settings" not in sys.modules:
    _settings_mod = ModuleType("llm_engineering.settings")
    _settings_mod.settings = _IntegrationSettings()
    sys.modules["llm_engineering.settings"] = _settings_mod


def _load_provider_module():
    """Load llm_provider.py bypassing __init__ chain."""
    spec = importlib.util.spec_from_file_location(
        "llm_engineering.infrastructure.llm_provider",
        "llm_engineering/infrastructure/llm_provider.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["llm_engineering.infrastructure.llm_provider"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_provider_module()


@pytest.fixture
def minimax_api_key():
    """Get MiniMax API key from environment."""
    key = os.environ.get("MINIMAX_API_KEY")
    if not key:
        pytest.skip("MINIMAX_API_KEY not set")
    return key


def test_minimax_chat_completion(minimax_api_key):
    """Test that MiniMax can generate a chat completion."""
    s = _IntegrationSettings(
        LLM_PROVIDER="minimax",
        MINIMAX_API_KEY=minimax_api_key,
        MINIMAX_MODEL_ID="MiniMax-M2.7",
    )
    with patch.object(_mod, "settings", s):
        model = _mod.get_chat_model(temperature=0.5)
        response = model.invoke("What is 2 + 2? Answer with just the number.")

        assert response.content is not None
        assert len(response.content.strip()) > 0
        assert "4" in response.content


def test_minimax_highspeed_model(minimax_api_key):
    """Test that MiniMax-M2.7-highspeed model works."""
    s = _IntegrationSettings(
        LLM_PROVIDER="minimax",
        MINIMAX_API_KEY=minimax_api_key,
        MINIMAX_MODEL_ID="MiniMax-M2.7-highspeed",
    )
    with patch.object(_mod, "settings", s):
        model = _mod.get_chat_model(temperature=0.5)
        response = model.invoke("Say hello in one word.")

        assert response.content is not None
        assert len(response.content.strip()) > 0


def test_minimax_with_max_tokens(minimax_api_key):
    """Test that max_tokens parameter is respected."""
    s = _IntegrationSettings(
        LLM_PROVIDER="minimax",
        MINIMAX_API_KEY=minimax_api_key,
        MINIMAX_MODEL_ID="MiniMax-M2.7",
    )
    with patch.object(_mod, "settings", s):
        model = _mod.get_chat_model(temperature=0.5, max_tokens=50)
        response = model.invoke("Write a short poem about AI.")

        assert response.content is not None
        assert len(response.content.strip()) > 0
