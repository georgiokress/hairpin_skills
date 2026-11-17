#!/usr/bin/env python3
"""Validate a Phase B SPI run manifest and its cross-field invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


DEFAULT_SCHEMA = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "run_manifest.schema.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--verify-files", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cross_fields(manifest: dict) -> list[str]:
    errors: list[str] = []
    process = manifest["process"]
    magnitude = float(process["normal_force_magnitude_n"])
    command = float(process["normal_force_command_n"])
    if abs(command + magnitude) > 1e-9:
        errors.append(
            "normal_force_command_n must equal -normal_force_magnitude_n"
        )

    purpose = manifest["identity"]["purpose"]
    split = manifest["doe"]["planned_split"]
    if purpose.startswith("validation_") and split != "validation":
        errors.append("validation purpose requires planned_split='validation'")
    training_purposes = {"qualification", "reference_training", "doe_training"}
    if purpose in training_purposes and split == "validation":
        errors.append(f"purpose {purpose!r} must not use validation split")

    outcome = manifest["outcome"]
    acquisition = manifest["acquisition"]
    if outcome["status"] == "COMPLETED":
        if acquisition["bag_path"] is None:
            errors.append("completed run requires acquisition.bag_path")
        if not acquisition["raw_data_immutable"]:
            errors.append("completed run requires raw_data_immutable=true")

    ground_truth = manifest["ground_truth"]
    if ground_truth["reference_frozen_before_detector_tuning"]:
        if ground_truth["reference_file"] is None:
            errors.append("frozen ground truth requires reference_file")
        if ground_truth["absolute_edge_x_R_m"] is None:
            errors.append("frozen ground truth requires absolute_edge_x_R_m")
    return errors


def verify_referenced_files(manifest: dict) -> list[str]:
    errors: list[str] = []
    checks = [
        (
            manifest["software"]["collection_config_path"],
            manifest["software"]["collection_config_sha256"],
            "collection config",
        ),
        (
            manifest["ground_truth"]["reference_file"],
            manifest["ground_truth"]["reference_sha256"],
            "ground truth",
        ),
    ]
    for path_text, expected, label in checks:
        if path_text is None:
            continue
        path = Path(path_text)
        if not path.exists():
            errors.append(f"{label} file does not exist: {path}")
            continue
        if expected is not None and sha256(path) != expected:
            errors.append(f"{label} SHA-256 mismatch: {path}")
    return errors


def main() -> None:
    args = parse_args()
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    schema_errors = sorted(validator.iter_errors(manifest), key=lambda item: list(item.path))
    errors = [
        f"schema {'.'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
        for error in schema_errors
    ]
    errors.extend(validate_cross_fields(manifest))
    if args.verify_files:
        errors.extend(verify_referenced_files(manifest))

    if errors:
        print("INVALID RUN MANIFEST")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"VALID RUN MANIFEST: {args.manifest}")


if __name__ == "__main__":
    main()
