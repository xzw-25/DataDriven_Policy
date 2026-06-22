#!/usr/bin/env python3
"""Convert a pickle file into a readable UTF-8 text dump."""

from __future__ import annotations

import argparse
import math
import dataclasses
import datetime as dt
import json
import pickle
import pprint
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None


def _to_builtin(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _to_builtin(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if pd is not None and isinstance(value, pd.DataFrame):
        return {
            "__type__": "pandas.DataFrame",
            "shape": list(value.shape),
            "columns": [str(column) for column in value.columns],
            "records": value.to_dict(orient="records"),
        }
    if pd is not None and isinstance(value, pd.Series):
        return {
            "__type__": "pandas.Series",
            "name": value.name,
            "records": value.to_dict(),
        }
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, set):
        return sorted((_to_builtin(item) for item in value), key=repr)
    return value


def _format_timestamp(value: Any) -> str:
    if isinstance(value, (int, float)) and math.isfinite(value):
        iso = dt.datetime.fromtimestamp(value, tz=dt.timezone.utc).isoformat()
        return f"{value:.6f} ({iso})"
    return str(value)


def _format_number(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in string_rows))
        for index, header in enumerate(headers)
    ]
    lines = [" | ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers))]
    lines.append("-+-".join("-" * width for width in widths))
    lines.extend(
        " | ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
        for row in string_rows
    )
    return lines


def _is_daily_manifest(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    schema = value.get("schema")
    return (
        isinstance(schema, Mapping)
        and str(schema.get("version", "")).startswith("ai_control_daily_manifest")
        and isinstance(value.get("records"), Sequence)
        and isinstance(value.get("clips"), Sequence)
    )


def _daily_manifest_lines(value: Mapping[str, Any]) -> list[str]:
    manifest_info = value.get("manifest_info", {})
    identity = manifest_info.get("identity", {}) if isinstance(manifest_info, Mapping) else {}
    version = manifest_info.get("version", {}) if isinstance(manifest_info, Mapping) else {}
    created = manifest_info.get("time", {}).get("created_at", "") if isinstance(manifest_info, Mapping) else ""
    records = value.get("records", [])
    groups = value.get("continuous_groups", [])
    clips = value.get("clips", [])
    summary = value.get("summary", {})
    total_record_frames = sum(
        int(record.get("frame_count", 0) or 0)
        for record in records
        if isinstance(record, Mapping)
    )

    lines = [
        "",
        "=" * 80,
        "Daily Manifest Summary",
        "=" * 80,
        f"daily_manifest_id: {identity.get('daily_manifest_id', '')}",
        f"vehicle_id: {identity.get('vehicle_id', '')}",
        f"service_date: {identity.get('service_date', '')}",
        f"schema_version: {value.get('schema', {}).get('version', '')}",
        f"generator_version: {version.get('generator_version', '')}",
        f"classification_rule_version: {version.get('classification_rule_version', '')}",
        f"quality_rule_version: {version.get('quality_rule_version', '')}",
        f"created_at: {created}",
    ]

    if isinstance(summary, Mapping):
        lines.extend(
            [
                "",
                "Summary",
                f"- record_count: {summary.get('record_count', len(records))}",
                f"- continuous_group_count: {summary.get('continuous_group_count', len(groups))}",
                f"- clip_count: {summary.get('clip_count', len(clips))}",
                f"- total_frame_count: {summary.get('total_frame_count', total_record_frames)}",
            ]
        )

    if records:
        record_rows = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            record_rows.append(
                [
                    record.get("bag_sequence_in_day", ""),
                    record.get("bag_id", ""),
                    record.get("continuous_group_id", ""),
                    record.get("frame_count", ""),
                    _format_timestamp(record.get("begin_timestamp")),
                    _format_timestamp(record.get("end_timestamp")),
                ]
            )
        lines.extend(["", "Records", *_table(["seq", "bag_id", "group", "frames", "begin", "end"], record_rows)])

    if groups:
        group_rows = []
        for group in groups:
            if not isinstance(group, Mapping):
                continue
            group_rows.append(
                [
                    group.get("continuous_group_id", ""),
                    group.get("bag_count", ""),
                    ", ".join(str(bag_id) for bag_id in group.get("bag_ids", [])),
                ]
            )
        lines.extend(["", "Continuous Groups", *_table(["group", "bag_count", "bag_ids"], group_rows)])

    if clips:
        domain_counter: Counter[str] = Counter()
        type_counter: Counter[str] = Counter()
        domain_duration: dict[str, float] = defaultdict(float)
        type_duration: dict[str, float] = defaultdict(float)
        multi_part_count = 0
        total_duration = 0.0
        total_clip_frames = 0
        clip_rows = []

        for clip in clips:
            if not isinstance(clip, Mapping):
                continue
            domain = str(clip.get("data_domain", ""))
            clip_type = str(clip.get("clip_type", ""))
            duration = float(clip.get("duration_sec", 0.0) or 0.0)
            frame_count = int(clip.get("frame_count", 0) or 0)
            parts = clip.get("parts", [])
            part_count = len(parts) if isinstance(parts, Sequence) else 0

            domain_counter[domain] += 1
            type_counter[clip_type] += 1
            domain_duration[domain] += duration
            type_duration[clip_type] += duration
            total_duration += duration
            total_clip_frames += frame_count
            multi_part_count += int(part_count > 1)

            clip_rows.append(
                [
                    clip.get("clip_sequence_in_group", ""),
                    clip.get("clip_id", ""),
                    clip.get("continuous_group_id", ""),
                    domain,
                    clip_type,
                    _format_number(duration),
                    frame_count,
                    part_count,
                ]
            )

        domain_rows = [
            [domain, count, _format_number(domain_duration[domain])]
            for domain, count in sorted(domain_counter.items())
        ]
        type_rows = [
            [clip_type, count, _format_number(type_duration[clip_type])]
            for clip_type, count in sorted(type_counter.items())
        ]
        lines.extend(
            [
                "",
                "Clip Totals",
                f"- clip_count: {len(clips)}",
                f"- total_duration_sec: {_format_number(total_duration)}",
                f"- total_frame_count: {total_clip_frames}",
                f"- multi_part_clip_count: {multi_part_count}",
                "",
                "Clip Count By Data Domain",
                *_table(["data_domain", "clip_count", "duration_sec"], domain_rows),
                "",
                "Clip Count By Type",
                *_table(["clip_type", "clip_count", "duration_sec"], type_rows),
                "",
                "Clips",
                *_table(
                    ["seq", "clip_id", "group", "domain", "type", "duration_sec", "frames", "parts"],
                    clip_rows,
                ),
            ]
        )

    return lines


def _summary_lines(source: Path, value: Any) -> list[str]:
    lines = [
        f"Source: {source}",
        f"Object type: {type(value).__module__}.{type(value).__name__}",
    ]
    if isinstance(value, Mapping):
        lines.append(f"Top-level keys: {list(value.keys())}")
        for key, item in value.items():
            if isinstance(item, Mapping):
                lines.append(f"- {key}: dict(len={len(item)})")
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                lines.append(f"- {key}: {type(item).__name__}(len={len(item)})")
            else:
                lines.append(f"- {key}: {type(item).__name__}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        lines.append(f"Length: {len(value)}")
        for index, item in enumerate(value[:10]):
            lines.append(f"- [{index}]: {type(item).__module__}.{type(item).__name__}")
    if _is_daily_manifest(value):
        lines.extend(_daily_manifest_lines(value))
    return lines


def convert_pickle_to_text(
    source: str | Path,
    output: str | Path | None = None,
    *,
    include_full_content: bool = True,
) -> Path:
    source_path = Path(source)
    output_path = Path(output) if output is not None else source_path.with_suffix(".txt")
    with source_path.open("rb") as file:
        value = pickle.load(file)

    lines = _summary_lines(source_path, value)
    if include_full_content:
        converted = _to_builtin(value)
        lines.extend(
            [
                "",
                "=" * 80,
                "Pretty Printed Content",
                "=" * 80,
            ]
        )
        try:
            body = json.dumps(converted, ensure_ascii=False, indent=2, sort_keys=False)
        except TypeError:
            body = pprint.pformat(converted, width=120, sort_dicts=False)
        lines.append(body)
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle_path", help="Path to the .pkl file.")
    parser.add_argument("--output", "-o", help="Output .txt path. Defaults to the input path with .txt suffix.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Write only the parsed summary and omit the full pretty-printed payload.",
    )
    args = parser.parse_args()

    output = convert_pickle_to_text(
        args.pickle_path,
        args.output,
        include_full_content=not args.summary_only,
    )
    print(f"output={output}")
    print(f"bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
