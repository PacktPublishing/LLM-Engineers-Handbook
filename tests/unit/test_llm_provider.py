"""Unit tests for the LLM provider factory.

Tests import the llm_provider module directly, mocking ChatOpenAI to avoid
dependency issues with the full llm_engineering package.
"""

import importlib
import importlib.util
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch, call

import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict


class _TestSettings(BaseSettings):
    """Minimal settings class for testing."""

    model_config = SettingsConfigDict(env_file=None)

    LLM_PROVIDER: str = "openai"
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
    _settings_mod.settings = _TestSettings()
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


def _s(**overrides):
    return _TestSettings(**overrides)


class TestGetChatModel:
    """Tests for get_chat_model() factory function."""

    def test_openai_provider_calls_chatopenai(self):
        s = _s(LLM_PROVIDER="openai", OPENAI_API_KEY="test-key", OPENAI_MODEL_ID="gpt-4o-mini")
        mock_cls = MagicMock()
        with patch.object(_mod, "settings", s), patch.object(_mod, "ChatOpenAI", mock_cls):
            _mod.get_chat_model(temperature=0.5)
            mock_cls.assert_called_once_with(
                model="gpt-4o-mini",
                api_key="test-key",
                temperature=0.5,
            )

    def test_minimax_provider_calls_chatopenai_with_base_url(self):
        s = _s(LLM_PROVIDER="minimax", MINIMAX_API_KEY="mm-key", MINIMAX_MODEL_ID="MiniMax-M2.7")
        mock_cls = MagicMock()
        with patch.object(_mod, "settings", s), patch.object(_mod, "ChatOpenAI", mock_cls):
            _mod.get_chat_model(temperature=0.5)
            mock_cls.assert_called_once_with(
                model="MiniMax-M2.7",
                api_key="mm-key",
                base_url="https://api.minimax.io/v1",
                temperature=0.5,
            )

    def test_minimax_temperature_zero_clamped(self):
        s = _s(LLM_PROVIDER="minimax", MINIMAX_API_KEY="key")
        mock_cls = MagicMock()
        with patch.object(_mod, "settings", s), patch.object(_mod, "ChatOpenAI", mock_cls):
            _mod.get_chat_model(temperature=0)
            _, kwargs = mock_cls.call_args
            assert kwargs["temperature"] == 0.01

    def test_minimax_temperature_negative_clamped(self):
        s = _s(LLM_PROVIDER="minimax", MINIMAX_API_KEY="key")
        mock_cls = MagicMock()
        with patch.object(_mod, "settings", s), patch.object(_mod, "ChatOpenAI", mock_cls):
            _mod.get_chat_model(temperature=-1)
            _, kwargs = mock_cls.call_args
            assert kwargs["temperature"] == 0.01

    def test_minimax_temperature_normal_preserved(self):
        s = _s(LLM_PROVIDER="minimax", MINIMAX_API_KEY="key")
        mock_cls = MagicMock()
        with patch.object(_mod, "settings", s), patch.object(_mod, "ChatOpenAI", mock_cls):
            _mod.get_chat_model(temperature=0.7)
            _, kwargs = mock_cls.call_args
            assert kwargs["temperature"] == 0.7

    def test_minimax_missing_api_key_raises(self):
        s = _s(LLM_PROVIDER="minimax", MINIMAX_API_KEY=None)
        with patch.object(_mod, "settings", s):
            with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
                _mod.get_chat_model()

    def test_minimax_highspeed_model(self):
        s = _s(LLM_PROVIDER="minimax", MINIMAX_API_KEY="key", MINIMAX_MODEL_ID="MiniMax-M2.7-highspeed")
        mock_cls = MagicMock()
        with patch.object(_mod, "settings", s), patch.object(_mod, "ChatOpenAI", mock_cls):
            _mod.get_chat_model()
            _, kwargs = mock_cls.call_args
            assert kwargs["model"] == "MiniMax-M2.7-highspeed"

    def test_kwargs_forwarded(self):
        s = _s(LLM_PROVIDER="minimax", MINIMAX_API_KEY="key")
        mock_cls = MagicMock()
        with patch.object(_mod, "settings", s), patch.object(_mod, "ChatOpenAI", mock_cls):
            _mod.get_chat_model(temperature=0.5, max_tokens=100)
            _, kwargs = mock_cls.call_args
            assert kwargs["max_tokens"] == 100

    def test_provider_case_insensitive(self):
        s = _s(LLM_PROVIDER="MiniMax", MINIMAX_API_KEY="key", MINIMAX_MODEL_ID="MiniMax-M2.7")
        mock_cls = MagicMock()
        with patch.object(_mod, "settings", s), patch.object(_mod, "ChatOpenAI", mock_cls):
            _mod.get_chat_model()
            _, kwargs = mock_cls.call_args
            assert kwargs["model"] == "MiniMax-M2.7"
            assert kwargs["base_url"] == "https://api.minimax.io/v1"

    def test_openai_no_base_url(self):
        """OpenAI provider should NOT set a base_url."""
        s = _s(LLM_PROVIDER="openai", OPENAI_API_KEY="key")
        mock_cls = MagicMock()
        with patch.object(_mod, "settings", s), patch.object(_mod, "ChatOpenAI", mock_cls):
            _mod.get_chat_model()
            _, kwargs = mock_cls.call_args
            assert "base_url" not in kwargs

    def test_openai_temperature_zero_not_clamped(self):
        """OpenAI provider should NOT clamp temperature=0."""
        s = _s(LLM_PROVIDER="openai", OPENAI_API_KEY="key")
        mock_cls = MagicMock()
        with patch.object(_mod, "settings", s), patch.object(_mod, "ChatOpenAI", mock_cls):
            _mod.get_chat_model(temperature=0)
            _, kwargs = mock_cls.call_args
            assert kwargs["temperature"] == 0


class TestGetMaxTokenWindow:
    """Tests for get_max_token_window() function."""

    def test_openai_gpt4o_mini(self):
        s = _s(LLM_PROVIDER="openai", OPENAI_MODEL_ID="gpt-4o-mini")
        with patch.object(_mod, "settings", s):
            assert _mod.get_max_token_window() == int(128000 * 0.90)

    def test_openai_gpt35_turbo(self):
        s = _s(LLM_PROVIDER="openai", OPENAI_MODEL_ID="gpt-3.5-turbo")
        with patch.object(_mod, "settings", s):
            assert _mod.get_max_token_window() == int(16385 * 0.90)

    def test_minimax_m27(self):
        s = _s(LLM_PROVIDER="minimax", MINIMAX_MODEL_ID="MiniMax-M2.7")
        with patch.object(_mod, "settings", s):
            assert _mod.get_max_token_window() == int(204800 * 0.90)

    def test_minimax_highspeed(self):
        s = _s(LLM_PROVIDER="minimax", MINIMAX_MODEL_ID="MiniMax-M2.7-highspeed")
        with patch.object(_mod, "settings", s):
            assert _mod.get_max_token_window() == int(204800 * 0.90)

    def test_unknown_model_default(self):
        s = _s(LLM_PROVIDER="openai", OPENAI_MODEL_ID="future-model")
        with patch.object(_mod, "settings", s):
            assert _mod.get_max_token_window() == int(128000 * 0.90)


class TestDefaults:
    """Tests for default settings values."""

    def test_default_provider(self):
        assert _TestSettings().LLM_PROVIDER == "openai"

    def test_default_minimax_model(self):
        assert _TestSettings().MINIMAX_MODEL_ID == "MiniMax-M2.7"

    def test_default_openai_model(self):
        assert _TestSettings().OPENAI_MODEL_ID == "gpt-4o-mini"
