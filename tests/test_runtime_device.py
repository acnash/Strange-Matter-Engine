from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from runtime_device import (
    load_checkpoint,
    python_executable,
    requested_device,
    resolve_torch_device,
)


class FakeCuda:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class FakeTorch:
    def __init__(self, cuda_available):
        self.cuda = FakeCuda(cuda_available)

    @staticmethod
    def device(name):
        return name


def test_auto_selects_cpu_without_cuda():
    assert resolve_torch_device(FakeTorch(False), "auto") == "cpu"


def test_auto_selects_cuda_when_available():
    assert resolve_torch_device(FakeTorch(True), "auto") == "cuda"


def test_explicit_cuda_fails_clearly_without_cuda():
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_torch_device(FakeTorch(False), "cuda")


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match="Unknown device"):
        requested_device("gpu")


def test_active_python_is_default():
    assert python_executable() == Path(sys.executable).resolve()


def test_checkpoint_loading_maps_tensors_to_selected_device(tmp_path):
    class FakeLoader:
        @staticmethod
        def load(path, map_location, weights_only):
            return path, map_location, weights_only

    path, device, weights_only = load_checkpoint(
        FakeLoader, tmp_path / "model.pt", "cpu"
    )
    assert path == tmp_path / "model.pt"
    assert device == "cpu"
    assert weights_only is False
