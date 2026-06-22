from __future__ import annotations

import pickle

from scripts.pickle_to_text import convert_pickle_to_text


def test_convert_daily_manifest_pickle_to_text(tmp_path):
    pickle_path = tmp_path / "manifest.pkl"
    output_path = tmp_path / "manifest.txt"
    payload = {
        "schema": {"version": "ai_control_daily_manifest_v1.0"},
        "manifest_info": {
            "identity": {
                "daily_manifest_id": "PP381_20260521_DM_v1",
                "vehicle_id": "PP381",
                "service_date": "20260521",
            },
            "version": {
                "generator_version": "ai_control_pkl",
                "classification_rule_version": "rule-a",
                "quality_rule_version": "rule-b",
            },
            "time": {"created_at": "2026-06-17T11:03:01+00:00"},
        },
        "records": [
            {
                "bag_sequence_in_day": 0,
                "bag_id": "bag-0",
                "continuous_group_id": "group-0",
                "frame_count": 3,
                "begin_timestamp": 1779346680.0,
                "end_timestamp": 1779346681.0,
            }
        ],
        "continuous_groups": [
            {"continuous_group_id": "group-0", "bag_count": 1, "bag_ids": ["bag-0"]}
        ],
        "clips": [
            {
                "clip_sequence_in_group": 0,
                "clip_id": "clip-0",
                "continuous_group_id": "group-0",
                "data_domain": "ad",
                "clip_type": "clean_ad",
                "duration_sec": 1.0,
                "frame_count": 3,
                "parts": [{"clip_part_id": "part-0"}],
            }
        ],
        "summary": {"record_count": 1, "continuous_group_count": 1, "clip_count": 1},
    }
    with pickle_path.open("wb") as file:
        pickle.dump(payload, file)

    result = convert_pickle_to_text(pickle_path, output_path, include_full_content=False)

    assert result == output_path
    text = output_path.read_text(encoding="utf-8")
    assert "Daily Manifest Summary" in text
    assert "daily_manifest_id: PP381_20260521_DM_v1" in text
    assert "- total_frame_count: 3" in text
    assert "Clip Count By Data Domain" in text
    assert "clean_ad" in text
    assert "Pretty Printed Content" not in text
