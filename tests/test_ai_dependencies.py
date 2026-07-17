"""Tests for glas.ai.dependencies."""

from __future__ import annotations

import builtins
from collections.abc import Iterator

import pytest

from glas.ai.dependencies import (
    AI_EXTRA_INSTALL_HINT,
    describe_missing_ai_packages,
    import_build_sam2,
    import_sam2_image_predictor,
    import_torch,
    import_ultralytics,
    missing_ai_packages,
)
from glas.exceptions import AIDependencyError


@pytest.fixture
def _block_import(monkeypatch: pytest.MonkeyPatch) -> Iterator[set[str]]:
    """Make ``import <name>`` raise ImportError for any module name in the returned set."""
    blocked: set[str] = set()
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name in blocked or name.split(".")[0] in blocked:
            raise ImportError(f"simulated missing module: {name}")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    yield blocked


class TestImportTorch:
    def test_returns_injected_module(self) -> None:
        sentinel = object()
        assert import_torch(sentinel) is sentinel

    def test_imports_real_torch_when_installed(self) -> None:
        module = import_torch()
        assert module.__name__ == "torch"

    def test_raises_ai_dependency_error_when_torch_missing(self, _block_import: set[str]) -> None:
        _block_import.add("torch")
        with pytest.raises(AIDependencyError, match="torch"):
            import_torch()


class TestImportUltralytics:
    def test_returns_injected_module(self) -> None:
        sentinel = object()
        assert import_ultralytics(sentinel) is sentinel

    def test_imports_real_ultralytics_when_installed(self) -> None:
        module = import_ultralytics()
        assert module.__name__ == "ultralytics"

    def test_raises_ai_dependency_error_when_missing(self, _block_import: set[str]) -> None:
        _block_import.add("ultralytics")
        with pytest.raises(AIDependencyError, match="ultralytics"):
            import_ultralytics()


class TestImportBuildSam2:
    def test_returns_injected_callable(self) -> None:
        sentinel = object()
        assert import_build_sam2(sentinel) is sentinel

    def test_imports_real_build_sam2_when_installed(self) -> None:
        build_sam2 = import_build_sam2()
        assert callable(build_sam2)

    def test_raises_ai_dependency_error_when_missing(self, _block_import: set[str]) -> None:
        _block_import.add("sam2")
        with pytest.raises(AIDependencyError, match="sam2"):
            import_build_sam2()


class TestImportSam2ImagePredictor:
    def test_returns_injected_class(self) -> None:
        sentinel = object()
        assert import_sam2_image_predictor(sentinel) is sentinel

    def test_imports_real_predictor_class_when_installed(self) -> None:
        predictor_cls = import_sam2_image_predictor()
        assert predictor_cls.__name__ == "SAM2ImagePredictor"

    def test_raises_ai_dependency_error_when_missing(self, _block_import: set[str]) -> None:
        _block_import.add("sam2")
        with pytest.raises(AIDependencyError, match="sam2"):
            import_sam2_image_predictor()


class TestMissingAiPackages:
    def test_empty_when_everything_installed(self) -> None:
        # torch, ultralytics, and sam2 are all installed in the dev/test
        # environment (pyproject.toml's [dev] extra), so this should find
        # nothing missing.
        assert missing_ai_packages() == []

    def test_reports_each_blocked_package(self, _block_import: set[str]) -> None:
        _block_import.update({"torch", "sam2"})
        assert missing_ai_packages() == ["torch", "sam2"]


class TestDescribeMissingAiPackages:
    def test_lists_every_missing_package(self) -> None:
        message = describe_missing_ai_packages(["torch", "ultralytics"])
        assert "torch" in message
        assert "ultralytics" in message
        assert AI_EXTRA_INSTALL_HINT in message

    def test_empty_list_still_mentions_install_hint(self) -> None:
        message = describe_missing_ai_packages([])
        assert AI_EXTRA_INSTALL_HINT in message
