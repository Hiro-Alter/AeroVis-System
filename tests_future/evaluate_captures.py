"""Prueba 2: evaluacion reproducible de las 280 capturas, sin GUI ni AirSim."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.model_loader import GestureClassifier, ModelCatalog
from src.inference.predictor import GesturePredictor
from src.utils.config_loader import load_app_config
from src.vision.pipeline import VisionPipeline


MODELS = ("resnet18", "mobilenetv3_small")
CONDITIONS = (("simple", "lit"), ("busy", "lit"), ("simple", "dim"), ("busy", "dim"))
WARMUP_COUNT = 10


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_samples(root, labels):
    data = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"samples"} or not isinstance(data["samples"], dict):
        raise ValueError("El JSON debe contener unicamente samples, indexado por nombre de imagen.")
    expected = {}
    for hand in ("right", "left"):
        for background, lighting in CONDITIONS:
            for gesture in labels:
                for repetition in range(1, 4):
                    stem = f"{hand}_{background}_{lighting}_{gesture}_{repetition:02d}"
                    paired = (background, lighting) == ("busy", "lit") and repetition <= 2
                    for occlusion in (("none", "partial") if paired else ("none",)):
                        filename = f"{stem}_{occlusion}.png"
                        expected[filename] = dict(file=f"{hand}/{filename}", hand=hand,
                                                  background=background, lighting=lighting,
                                                  gesture=gesture, repetition=repetition,
                                                  occlusion=occlusion, pair_id=stem if paired else None)
    if data["samples"] != expected:
        raise ValueError("Las anotaciones no coinciden con las 280 capturas del protocolo. Revisar antes de evaluar.")
    actual = {p.relative_to(root).as_posix() for hand in ("right", "left") for p in (root / hand).glob("*.png")}
    if actual != {s["file"] for s in expected.values()}:
        raise ValueError("Faltan PNG o hay PNG sin anotar en las carpetas de manos.")
    samples = []
    for filename, sample in expected.items():
        path = root / sample["file"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Una imagen enlaza fuera del dataset.")
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Imagen ilegible: {filename}")
        samples.append({"filename": filename, **sample, "image_sha256": sha256(path)})
    return samples


def process_image(frame, vision, predictor, mirror):
    start = time.perf_counter()
    if mirror:
        frame = cv2.flip(frame, 1)
    result = vision.process(frame)
    prediction = predictor.predict_roi(result.roi.image_bgr) if result.roi is not None else None
    elapsed = (time.perf_counter() - start) * 1000.0
    return result, prediction, elapsed


def mean_std(values):
    return {"n": len(values), "mean_ms": statistics.mean(values) if values else None,
            "std_ms": statistics.stdev(values) if len(values) > 1 else None}


def classification(rows, labels):
    valid = [r for r in rows if r["roi_valid"]]
    matrix = [[0 for _ in labels] for _ in labels]
    index = {label: i for i, label in enumerate(labels)}
    for row in valid:
        matrix[index[row["gesture"]]][index[row["prediction"]]] += 1
    f1 = []
    for i in range(len(labels)):
        tp = matrix[i][i]
        fn = sum(matrix[i]) - tp
        fp = sum(line[i] for line in matrix) - tp
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"n": len(valid), "correct": sum(matrix[i][i] for i in range(len(labels))),
            "accuracy": sum(matrix[i][i] for i in range(len(labels))) / len(valid) if valid else None,
            "f1_macro": statistics.mean(f1) if valid else None,
            "missing_true_classes": [labels[i] for i, line in enumerate(matrix) if not sum(line)],
            "confusion_counts": matrix}


def group_rows(rows):
    groups = {"base_all": [r for r in rows if r["occlusion"] == "none"]}
    for background, lighting in CONDITIONS:
        groups[f"base_{background}_{lighting}"] = [r for r in groups["base_all"]
                                                    if (r["background"], r["lighting"]) == (background, lighting)]
    groups["paired_base_all"] = [r for r in rows if r["pair_id"] and r["occlusion"] == "none"]
    groups["partial_all"] = [r for r in rows if r["occlusion"] == "partial"]
    by_pair = {}
    for row in rows:
        if row["pair_id"]:
            by_pair.setdefault(row["pair_id"], []).append(row)
    complete = {key for key, pair in by_pair.items() if len(pair) == 2 and all(r["roi_valid"] for r in pair)}
    groups["paired_base_common_roi"] = [r for r in groups["paired_base_all"] if r["pair_id"] in complete]
    groups["partial_common_roi"] = [r for r in groups["partial_all"] if r["pair_id"] in complete]
    return groups


def summarize(rows, labels):
    detection, recognition, timing = [], [], []
    matrices = {}
    reference = {r["filename"]: r for r in rows if r["model"] == MODELS[0]}
    for model in MODELS:
        model_rows = [r for r in rows if r["model"] == model]
        if {r["filename"] for r in model_rows} != set(reference) or len(model_rows) != len(reference):
            raise ValueError("Los modelos no evaluaron las mismas imagenes.")
        for row in model_rows:
            base = reference[row["filename"]]
            for field in ("image_sha256", "hand_detected", "roi_valid", "roi_sha256", "roi_box"):
                if row[field] != base[field]:
                    raise ValueError(f"ROI/deteccion diferente entre modelos: {row['filename']} ({field}).")
        for group, selected in group_rows(model_rows).items():
            if model == MODELS[0] and not group.endswith("common_roi"):
                detected = sum(r["hand_detected"] for r in selected)
                detection.append(dict(group=group, n=len(selected), detected=detected,
                                      missed=len(selected) - detected,
                                      roi_valid=sum(r["roi_valid"] for r in selected),
                                      sensitivity=detected / len(selected) if selected else None))
            scores = classification(selected, labels)
            matrices[f"{model}/{group}"] = scores.pop("confusion_counts")
            recognition.append(dict(model=model, group=group, n_input=len(selected),
                                    no_roi=sum(not r["roi_valid"] for r in selected), **scores))
            valid = [r for r in selected if r["roi_valid"]]
            missing = [r for r in selected if not r["roi_valid"]]
            for stage, values in (("classifier", [r["inference_ms"] for r in valid]),
                                  ("processing_with_roi", [r["processing_ms"] for r in valid]),
                                  ("processing_without_roi", [r["processing_ms"] for r in missing])):
                timing.append(dict(model=model, group=group, stage=stage, **mean_std(values)))
    return dict(detection=detection, classification=recognition, timing=timing,
                confusion_labels=labels, confusion_counts=matrices)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def review_tile(sample, result):
    tile = np.full((270, 640, 3), 245, dtype=np.uint8)
    tile[:225, :400] = cv2.resize(result.annotated_bgr, (400, 225))
    if result.roi is not None:
        tile[:224, 408:632] = cv2.resize(result.roi.image_bgr, (224, 224))
    cv2.putText(tile, sample["filename"], (4, 242), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1)
    status = "DETECTED / ROI" if result.roi is not None else "NO ROI"
    cv2.putText(tile, status, (4, 261), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return tile


def save_review_page(directory, tiles, page):
    canvas = np.full((270 * 4, 640 * 3, 3), 245, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        y, x = divmod(i, 3)
        canvas[y * 270:(y + 1) * 270, x * 640:(x + 1) * 640] = tile
    if not cv2.imwrite(str(directory / f"page_{page:02d}.jpg"), canvas):
        raise OSError("No se pudo guardar una hoja de revision.")


def pilot_rois(directory, config):
    rois, records = [], []
    for path in sorted(directory.glob("*.jpg")):
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Piloto ilegible: {path.name}")
        if config.get("camera", {}).get("mirror", True):
            frame = cv2.flip(frame, 1)
        vision = VisionPipeline(config.get("vision", {}))
        try:
            result = vision.process(frame)
            records.append(dict(file=path.name, sha256=sha256(path), detected=result.hand_detected,
                                roi_valid=result.roi is not None))
            if result.roi is not None:
                rois.append(result.roi.image_bgr.copy())
        finally:
            vision.close()
    if not rois:
        raise ValueError("El piloto separado no produjo ninguna ROI; no se inicio la evaluacion formal.")
    return rois, records


def evaluate(args):
    config = load_app_config(args.config)
    catalog = ModelCatalog(config["models"]["manifest"])
    labels = catalog.get(MODELS[0]).labels
    if len(labels) != 10 or len(set(labels)) != 10 or catalog.get(MODELS[1]).labels != labels:
        raise ValueError("Se requieren los dos modelos con las mismas diez clases, en el mismo orden.")
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("La carpeta de resultados ya existe; usa una nueva para conservar la corrida anterior.")
    if output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError("La carpeta de resultados debe estar separada del dataset.")
    if args.pilot_dir.resolve().is_relative_to(dataset):
        raise ValueError("El piloto debe ser externo a las capturas formales.")
    samples = load_samples(dataset, labels)
    annotation_hash = sha256(dataset / "annotations.json")
    rois, pilot = pilot_rois(args.pilot_dir, config)
    if {p["sha256"] for p in pilot} & {s["image_sha256"] for s in samples}:
        raise ValueError("El piloto contiene imagenes del conjunto formal.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "review").mkdir()
    sources = sorted(set(ROOT.glob("src/vision/*.py")) | set(ROOT.glob("src/inference/*.py"))
                     | {Path(__file__).resolve()})
    info = dict(status="incomplete", samples=len(samples), model_order=list(MODELS),
                annotations_sha256=annotation_hash, config_sha256=sha256(args.config),
                config={k: config[k] for k in ("camera", "vision", "models", "inference")},
                source_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in sources},
                versions={p: importlib.metadata.version(p) for p in
                          ("torch", "torch-directml", "mediapipe", "opencv-python", "numpy")},
                python=platform.python_version(), system=platform.platform(), cpu=platform.processor(),
                torch_threads=torch.get_num_threads(), opencv_threads=cv2.getNumThreads(),
                pilot=pilot, warmup_per_model=WARMUP_COUNT, models={},
                protocol={"independent_photos": True, "detector": "new VisionPipeline per image, outside timer",
                          "mirror": config.get("camera", {}).get("mirror", True), "batch_size": 1,
                          "classifier_time": "GesturePredictor timer: forward, softmax, argmax and CPU scalar reads; excludes preprocessing",
                          "processing_time": "memory image, mirror, VisionPipeline including overlay, preprocess and prediction",
                          "excluded_time": "disk I/O, model load, detector creation/close, pilot, warmup, review and result writing",
                          "std": "sample standard deviation (n-1); null for n < 2",
                          "f1_macro": "fixed ten labels, zero_division=0; null for no valid ROI",
                          "confidence_filter": False, "visual_review": "pending"})
    write_json(output / "run_info.json", info)
    rows = []
    raw_handle = None
    try:
        for model in MODELS:
            definition = catalog.get(model)
            classifier = GestureClassifier.load(definition, config["inference"].get("preferred_device", "directml"),
                                                config["inference"].get("fallback_device", "cpu"))
            predictor = GesturePredictor(classifier)
            info["models"][model] = dict(device=classifier.device.name, device_message=classifier.device.message,
                                         torch_device=str(classifier.device.torch_device),
                                         weights_sha256=sha256(definition.torchscript_path),
                                         metadata_sha256=sha256(definition.metadata_path),
                                         labels_sha256=sha256(definition.labels_path))
            if classifier.device.name == "directml":
                import torch_directml
                info["models"][model]["adapter"] = torch_directml.device_name(classifier.device.torch_device.index or 0)
            write_json(output / "run_info.json", info)
            print(f"{model}: {classifier.device.name}; piloto y {WARMUP_COUNT} calentamientos", flush=True)
            for i in range(WARMUP_COUNT):
                predictor.predict_roi(rois[i % len(rois)])
            tiles, page = [], 1
            for i, sample in enumerate(samples, 1):
                path = dataset / sample["file"]
                if sha256(path) != sample["image_sha256"]:
                    raise ValueError("Una imagen cambio durante la evaluacion.")
                frame = cv2.imread(str(path))
                if frame is None:
                    raise ValueError(f"Imagen ilegible: {sample['filename']}")
                vision = VisionPipeline(config.get("vision", {}))
                try:
                    result, prediction, elapsed = process_image(frame, vision, predictor, info["protocol"]["mirror"])
                finally:
                    vision.close()
                row = {**sample, "model": model, "device": classifier.device.name,
                       "hand_detected": result.hand_detected, "roi_valid": result.roi is not None,
                       "roi_box": list(result.roi.box) if result.roi is not None else None,
                       "roi_sha256": hashlib.sha256(result.roi.image_bgr.tobytes()).hexdigest() if result.roi is not None else None,
                       "prediction": prediction.gesture if prediction else None,
                       "confidence": prediction.confidence if prediction else None,
                       "correct": prediction.gesture == sample["gesture"] if prediction else None,
                       "inference_ms": prediction.inference_time_ms if prediction else None,
                       "processing_ms": elapsed, "error": result.error}
                if not rows:
                    raw_handle = (output / "predictions.csv").open("w", newline="", encoding="utf-8-sig")
                    raw_writer = csv.DictWriter(raw_handle, fieldnames=list(row))
                    raw_writer.writeheader()
                raw_writer.writerow(row)
                raw_handle.flush()
                rows.append(row)
                if model == MODELS[0]:
                    tiles.append(review_tile(sample, result))
                    if len(tiles) == 12 or i == len(samples):
                        save_review_page(output / "review", tiles, page)
                        tiles, page = [], page + 1
                if i % 20 == 0 or i == len(samples):
                    print(f"  {i}/{len(samples)}", flush=True)
            del predictor, classifier
        summary = summarize(rows, labels)
        if sha256(dataset / "annotations.json") != annotation_hash:
            raise ValueError("Las anotaciones cambiaron durante la evaluacion.")
        for sample in samples:
            if sha256(dataset / sample["file"]) != sample["image_sha256"]:
                raise ValueError("Una captura cambio antes de finalizar la evaluacion.")
        for source, digest in info["source_sha256"].items():
            if sha256(ROOT / source) != digest:
                raise ValueError("El codigo cambio durante la evaluacion.")
        for model in MODELS:
            if sha256(catalog.get(model).torchscript_path) != info["models"][model]["weights_sha256"]:
                raise ValueError("Un modelo cambio durante la evaluacion.")
        if sha256(args.config) != info["config_sha256"]:
            raise ValueError("La configuracion cambio durante la evaluacion.")
        write_json(output / "summary.json", summary)
        for name in ("detection", "classification", "timing"):
            write_csv(output / f"{name}.csv", summary[name])
        write_csv(output / "visual_review.csv", [dict(filename=s["filename"], label_ok="", detection_ok="", note="") for s in samples])
        info["status"] = "complete"
        write_json(output / "run_info.json", info)
        print(f"Prueba completada: {output}. Revision visual pendiente.", flush=True)
    except BaseException as exc:
        info["status"] = "failed"
        info["error"] = f"{type(exc).__name__}: {exc}"
        write_json(output / "run_info.json", info)
        raise
    finally:
        if raw_handle is not None:
            raw_handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Carpeta nueva, separada del dataset")
    parser.add_argument("--config", type=Path, default=ROOT / "config/app_config.json")
    parser.add_argument("--pilot-dir", type=Path, default=ROOT / "tests/assets/hand_samples")
    args = parser.parse_args()
    try:
        evaluate(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
