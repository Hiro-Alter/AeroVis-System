from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.device import SelectedDevice, select_device
from src.inference.model_loader import GestureClassifier, ModelCatalog, _load_torchscript


def main() -> int:
    catalog = ModelCatalog("models/manifest.json")
    definition = catalog.get("resnet18")
    simulated_device = SelectedDevice("directml", torch.device("cpu"), "DirectML simulado")
    cpu_module = _load_torchscript(definition.torchscript_path, torch.device("cpu"))
    # Un fallo de carga debe conservar la causa al volver a CPU.
    with patch("src.inference.model_loader.select_device", return_value=simulated_device), patch(
        "src.inference.model_loader._load_torchscript",
        side_effect=[AssertionError("fallo de prueba"), cpu_module],
    ):
        fallback = GestureClassifier.load(definition)
    assert fallback.device.name == "cpu"
    assert "AssertionError: fallo de prueba" in fallback.device.message

    selected = select_device()
    if selected.name != "directml":
        print(f"SKIP prueba GPU: {selected.message}; respaldo CPU OK")
        return 0

    generator = torch.Generator().manual_seed(42)
    for name in catalog.model_names:
        definition = catalog.get(name)
        cpu = GestureClassifier.load(definition, preferred_device="cpu")
        gpu = GestureClassifier.load(definition)
        assert gpu.device.name == "directml", gpu.device.message
        assert all(p.device == selected.torch_device for p in gpu.module.parameters())
        assert all(b.device == selected.torch_device for b in gpu.module.buffers())
        size = definition.input_spec.image_size
        tensor = torch.randn(1, 3, size, size, generator=generator)
        gpu_tensor = tensor.to(gpu.device.torch_device)
        with torch.no_grad():
            expected = cpu.module(tensor)
            actual = gpu.module(gpu_tensor).cpu()
        # La GPU debe mantener los resultados dentro del error numerico esperado.
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
        cpu_prediction = cpu.predict(tensor)
        gpu_prediction = gpu.predict(gpu_tensor)
        assert gpu_prediction[0] == cpu_prediction[0]
        assert abs(gpu_prediction[1] - cpu_prediction[1]) < 1e-3
        print(f"{name}: DirectML OK, prediccion coincide con CPU")

    print("smoke_test_directml ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
