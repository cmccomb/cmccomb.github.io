"""Tests for the SPECTER2 sentence-transformer construction helpers."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


class _DummyTransformer:
    """Minimal stand-in for the sentence-transformers Transformer class."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.auto_model = None

    @staticmethod
    def get_word_embedding_dimension() -> int:
        return 10


class _DummyPooling:
    def __init__(self, *_: object, **__: object) -> None:  # pragma: no cover - trivial
        return


class _DummyNormalize:
    def __call__(self, *_: object, **__: object) -> None:  # pragma: no cover - trivial
        return


class _DummySentenceTransformer:
    """Collect the modules passed to the SentenceTransformer constructor."""

    def __init__(self, *, modules: list[object]) -> None:
        self.modules = modules


class _DummyAdapterModel:
    def __init__(self, base_path: str) -> None:
        self.base_path = base_path
        self.active_adapter = None
        self.adapters_loaded: list[tuple[str, str, str]] = []

    @classmethod
    def from_pretrained(cls, base_path: str) -> "_DummyAdapterModel":
        return cls(base_path)

    def load_adapter(self, adapter_path: str, *, source: str, load_as: str) -> str:
        self.adapters_loaded.append((adapter_path, source, load_as))
        return f"adapter::{load_as}"

    def set_active_adapters(self, adapter_name: str) -> None:
        self.active_adapter = adapter_name


@pytest.fixture(autouse=True)
def _cleanup_modules() -> None:
    """Ensure stubbed modules do not leak between tests."""

    original_modules = sys.modules.copy()
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name not in original_modules:
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)


def test_build_specter2_sentence_transformer_uses_local_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    """The builder should rely on locally cached model artefacts when offline."""

    dummy_sentence_transformers = ModuleType("sentence_transformers")
    dummy_models = ModuleType("sentence_transformers.models")
    dummy_sentence_transformers.models = dummy_models
    dummy_models.Transformer = _DummyTransformer
    dummy_models.Pooling = _DummyPooling
    dummy_models.Normalize = _DummyNormalize

    def _sentence_transformer_constructor(*, modules: list[object]) -> _DummySentenceTransformer:
        return _DummySentenceTransformer(modules=modules)

    dummy_sentence_transformers.SentenceTransformer = _sentence_transformer_constructor

    sys.modules["sentence_transformers"] = dummy_sentence_transformers
    sys.modules["sentence_transformers.models"] = dummy_models

    adapters_module = ModuleType("adapters")
    adapters_module.AutoAdapterModel = _DummyAdapterModel
    sys.modules["adapters"] = adapters_module

    downloads: dict[str, bool] = {}

    def _fake_snapshot_download(repo_id: str, *, local_files_only: bool) -> str:
        downloads[repo_id] = local_files_only
        return f"/cache/{repo_id.replace('/', '_')}"

    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setitem(sys.modules, "huggingface_hub", ModuleType("huggingface_hub"))
    sys.modules["huggingface_hub"].snapshot_download = _fake_snapshot_download

    module = importlib.import_module("_scripts.build_json")
    importlib.reload(module)

    model = module._build_specter2_sentence_transformer()

    assert downloads == {
        module.SPECTER2_BASE_MODEL_NAME: True,
        module.KEYBERT_MODEL_NAME: True,
    }

    assert isinstance(model, _DummySentenceTransformer)
    transformer_module = model.modules[0]
    assert isinstance(transformer_module, _DummyTransformer)
    assert transformer_module.model_path == "/cache/allenai_specter2_base"

    adapter_model = transformer_module.auto_model
    assert isinstance(adapter_model, _DummyAdapterModel)
    assert adapter_model.base_path == "/cache/allenai_specter2_base"
    assert adapter_model.adapters_loaded == [
        ("/cache/allenai_specter2", "local", module.KEYBERT_MODEL_NAME)
    ]

