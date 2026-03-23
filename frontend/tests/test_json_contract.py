import json
from pathlib import Path
import sys

import pandas as pd
import pytest

FRONTEND_DIR = Path(__file__).resolve().parents[1]
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

from services import export_project_json, load_project_from_json


class _UploadedJSON:
    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._data


def _base_payload() -> dict:
    return {
        "project_name": "Contract Test",
        "description": "Project description for contract test.",
        "criteria": {
            "price": {"type": "cardinal", "shape": "cost", "n_segments": 2},
        },
        "algorithm_settings": {
            "algorithm": "UTASTAR",
            "sigma": 0.001,
            "missing_value_treatment": "assumeAverageValue",
        },
        "alternatives": {
            "reference_alternatives": [
                {"Name": "A", "price": 10.0, "rank": 1},
                {"Name": "B", "price": 12.0, "rank": 2},
            ],
            "non_reference_alternatives": [
                {"Name": "C", "price": 7.0},
            ],
        },
    }


def test_load_project_accepts_embedded_rank_contract():
    payload = _base_payload()

    (
        project_name,
        project_description,
        criteria_defs,
        alternatives_df,
        rankings,
        reference_names,
        loaded_results,
        algorithm_settings,
    ) = load_project_from_json(_UploadedJSON(payload))

    assert project_name == "Contract Test"
    assert project_description == "Project description for contract test."
    assert list(criteria_defs.keys()) == ["price"]
    assert list(alternatives_df["Name"]) == ["A", "B", "C"]
    assert rankings == [1, 2]
    assert reference_names == ["A", "B"]
    assert loaded_results is None
    assert algorithm_settings["algorithm"] == "UTASTAR"


def test_load_project_rejects_missing_reference_rank():
    payload = _base_payload()
    del payload["alternatives"]["reference_alternatives"][0]["rank"]

    with pytest.raises(ValueError, match="must include integer `rank`"):
        load_project_from_json(_UploadedJSON(payload))


def test_load_project_rejects_reserved_criterion_name_rank():
    payload = _base_payload()
    payload["criteria"] = {"rank": {"type": "cardinal", "shape": "gain", "n_segments": 2}}
    payload["alternatives"]["reference_alternatives"] = [
        {"Name": "A", "rank": 1},
        {"Name": "B", "rank": 2},
    ]
    payload["alternatives"]["non_reference_alternatives"] = [{"Name": "C"}]

    with pytest.raises(ValueError, match="Criterion name `rank` is reserved"):
        load_project_from_json(_UploadedJSON(payload))


def test_export_embeds_reference_rank_and_omits_legacy_fields():
    alternatives_df = pd.DataFrame(
        [
            {"Name": "A", "price": 10.0},
            {"Name": "B", "price": 12.0},
            {"Name": "C", "price": 7.0},
        ]
    )

    payload_json = export_project_json(
        project_name="Export Test",
        project_description="Export description",
        criteria_defs={"price": {"type": "cardinal", "shape": "cost", "n_segments": 2}},
        alternatives_df=alternatives_df,
        rankings=[2, 1],
        results=None,
        algorithm_settings={
            "algorithm": "UTASTAR",
            "sigma": 0.001,
            "missing_value_treatment": "assumeAverageValue",
            "reference_names": ["A", "B"],
            "output_folder": "./ignored",
        },
        reference_names=["A", "B"],
    )

    payload = json.loads(payload_json)

    assert "rankings" not in payload
    assert payload["description"] == "Export description"
    assert "reference_names" not in payload["algorithm_settings"]
    assert "output_folder" not in payload["algorithm_settings"]

    ref_rows = payload["alternatives"]["reference_alternatives"]
    non_ref_rows = payload["alternatives"]["non_reference_alternatives"]

    assert [row["Name"] for row in ref_rows] == ["A", "B"]
    assert [row["rank"] for row in ref_rows] == [2, 1]
    assert all("rank" not in row for row in non_ref_rows)


def test_export_rejects_reserved_criterion_name_rank():
    alternatives_df = pd.DataFrame(
        [
            {"Name": "A", "rank": 10.0},
            {"Name": "B", "rank": 12.0},
        ]
    )

    with pytest.raises(ValueError, match="Criterion name `rank` is reserved"):
        export_project_json(
            project_name="Export Invalid",
            project_description="Invalid",
            criteria_defs={"rank": {"type": "cardinal", "shape": "gain", "n_segments": 2}},
            alternatives_df=alternatives_df,
            rankings=[1, 2],
            results=None,
            algorithm_settings={"algorithm": "UTASTAR", "sigma": 0.001},
            reference_names=["A", "B"],
        )


def test_export_import_round_trip_preserves_reference_ranks():
    alternatives_df = pd.DataFrame(
        [
            {"Name": "A", "price": 10.0},
            {"Name": "B", "price": 12.0},
            {"Name": "C", "price": 7.0},
        ]
    )

    exported = export_project_json(
        project_name="Round Trip",
        project_description="Round trip description",
        criteria_defs={"price": {"type": "cardinal", "shape": "cost", "n_segments": 2}},
        alternatives_df=alternatives_df,
        rankings=[2, 1],
        results=None,
        algorithm_settings={"algorithm": "UTASTAR", "sigma": 0.001},
        reference_names=["A", "B"],
    )

    (
        _project_name,
        loaded_description,
        _criteria_defs,
        _alternatives_df,
        rankings,
        reference_names,
        _loaded_results,
        _algorithm_settings,
    ) = load_project_from_json(_UploadedJSON(json.loads(exported)))

    assert reference_names == ["A", "B"]
    assert rankings == [2, 1]
    assert loaded_description == "Round trip description"
