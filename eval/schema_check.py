"""Validates every trace line against telemetry/schema.json (G6)."""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "streaming_rag" / "telemetry" / "schema.json"


def load_schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_events(events: list[dict], schema: dict | None = None) -> list[str]:
    schema = schema or load_schema()
    validator = jsonschema.Draft7Validator(schema)
    errors = []
    for i, event in enumerate(events):
        for err in validator.iter_errors(event):
            errors.append(f"event[{i}] ({event.get('event')}): {err.message}")
    return errors
