"""Frontend services: loading data and running UTA analysis."""

from __future__ import annotations

import io
import json
import inspect
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

from uta_solver import UTAEstimator
from uta_solver.criteria import CardinalCriterion, OrdinalCriterion, NominalCriterion, Criterion

from examples import load_apartments, load_cars


REQUIRED_JSON_KEYS = ["project_name", "criteria", "alternatives"]
RESERVED_CRITERION_NAMES = {"rank"}

DEFAULT_ALGORITHM_SETTINGS = {
    "algorithm": "UTASTAR",
    "sigma": 0.001,
    "breakpoints": "quantile",
    "theta": 1.0,
    "big_m": 1000.0,
    "ineq": 0.001,
    "max_nonmonotonicity_degree": 2,
    "objective_threshold": 0.01,
    "minimum_improvement": 0.0,
    "missing_value_treatment": "assumeAverageValue",
}


def expected_alternative_columns(criteria_defs: Dict[str, dict]) -> List[str]:
    return ["Name"] + list(criteria_defs.keys())


def _validate_reserved_criterion_names(criteria_defs: Dict[str, dict]) -> None:
    conflicts = [
        str(name)
        for name in criteria_defs.keys()
        if str(name).strip().lower() in RESERVED_CRITERION_NAMES
    ]
    if conflicts:
        joined = ", ".join(f"`{name}`" for name in conflicts)
        raise ValueError(
            "Criterion name `rank` is reserved for reference ranking metadata in JSON. "
            f"Please rename/remove conflicting criterion(s): {joined}."
        )


def validate_alternatives_schema(alternatives_df: pd.DataFrame, criteria_defs: Dict[str, dict]) -> None:
    expected = set(expected_alternative_columns(criteria_defs))
    actual = set(alternatives_df.columns)

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing columns: {', '.join(missing)}")
        if extra:
            parts.append(f"unexpected columns: {', '.join(extra)}")
        raise ValueError(f"Alternatives schema mismatch ({'; '.join(parts)}).")


def get_missing_required_values(alternatives_df: pd.DataFrame, criteria_defs: Dict[str, dict]) -> Dict[str, int]:
    missing_counts: Dict[str, int] = {}
    if alternatives_df is None or alternatives_df.empty:
        return missing_counts

    for criterion_name, spec in criteria_defs.items():
        if criterion_name not in alternatives_df.columns:
            missing_counts[criterion_name] = len(alternatives_df)
            continue

        series = alternatives_df[criterion_name]
        crit_type = spec.get("type")
        if crit_type == "cardinal":
            numeric = pd.to_numeric(series, errors="coerce")
            missing_mask = numeric.isna()
        else:
            as_text = series.astype("string")
            blank_mask = as_text.str.strip().eq("")
            missing_mask = series.isna() | blank_mask

            categories = spec.get("categories") or []
            if categories:
                invalid_mask = (~missing_mask) & (~as_text.isin(categories))
                missing_mask = missing_mask | invalid_mask

        count = int(missing_mask.sum())
        if count > 0:
            missing_counts[criterion_name] = count

    return missing_counts


def _normalize_name_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Name" not in df.columns and "name" in df.columns:
        return df.rename(columns={"name": "Name"})
    if "Name" not in df.columns:
        if len(df.columns) > 1:
            return df.rename(columns={df.columns[0]: "Name"})
        df = df.copy()
        df.insert(0, "Name", [f"Alt{i+1}" for i in range(len(df))])
    return df


def _deserialize_results(results_payload: Any) -> dict | None:
    if not isinstance(results_payload, dict):
        return None

    required = {"objective_value", "kendall_tau", "marginal_utilities"}
    if not required.issubset(results_payload.keys()):
        return None

    utilities_df_payload = results_payload.get("utilities_df")
    utilities_df = None
    if isinstance(utilities_df_payload, list):
        utilities_df = pd.DataFrame(utilities_df_payload)

    try:
        objective_value = float(results_payload["objective_value"])
        kendall_tau = float(results_payload["kendall_tau"])
    except (TypeError, ValueError):
        return None

    return {
        "model": None,
        "utilities_df": utilities_df,
        "objective_value": objective_value,
        "kendall_tau": kendall_tau,
        "marginal_utilities": results_payload["marginal_utilities"],
        "breakpoint_utilities": results_payload.get("breakpoint_utilities"),
        "partial_values": results_payload.get("partial_values"),
        "criteria": None,
        "is_loaded": True,
    }


def _normalize_algorithm_settings(settings_payload: Any) -> dict:
    normalized = dict(DEFAULT_ALGORITHM_SETTINGS)

    if isinstance(settings_payload, dict):
        if "algorithm" in settings_payload:
            algorithm = str(settings_payload.get("algorithm", "UTASTAR")).upper()
            normalized["algorithm"] = algorithm if algorithm in {"UTASTAR", "UTANM"} else "UTASTAR"

        if "breakpoints" in settings_payload:
            breakpoints_mode = str(settings_payload.get("breakpoints", "quantile")).lower()
            if breakpoints_mode in {"quantile", "uniform"}:
                normalized["breakpoints"] = breakpoints_mode

        for key in ("sigma", "theta", "big_m", "ineq", "objective_threshold", "minimum_improvement"):
            if key in settings_payload:
                try:
                    normalized[key] = float(settings_payload[key])
                except (TypeError, ValueError):
                    pass

        if "max_nonmonotonicity_degree" in settings_payload:
            try:
                normalized["max_nonmonotonicity_degree"] = int(settings_payload["max_nonmonotonicity_degree"])
            except (TypeError, ValueError):
                pass

        if "missing_value_treatment" in settings_payload:
            mvt = str(settings_payload.get("missing_value_treatment", "assumeAverageValue"))
            if mvt in {"assumeAverageValue", "assumeZeroValue"}:
                normalized["missing_value_treatment"] = mvt

    return normalized


def _serialize_algorithm_settings_for_json(settings_payload: Optional[dict]) -> dict:
    normalized = _normalize_algorithm_settings(settings_payload)
    algorithm = str(normalized.get("algorithm", "UTASTAR")).upper()

    common = {
        "algorithm": algorithm,
        "sigma": float(normalized["sigma"]),
        "breakpoints": str(normalized["breakpoints"]),
        "missing_value_treatment": str(normalized["missing_value_treatment"]),
    }

    if algorithm == "UTANM":
        return {
            **common,
            "theta": float(normalized["theta"]),
            "big_m": float(normalized["big_m"]),
            "ineq": float(normalized["ineq"]),
            "max_nonmonotonicity_degree": int(normalized["max_nonmonotonicity_degree"]),
            "objective_threshold": float(normalized["objective_threshold"]),
            "minimum_improvement": float(normalized["minimum_improvement"]),
        }

    return common


def criteria_defs_to_objects(
    criteria_defs: Dict[str, dict],
    feature_order: List[str],
) -> List[Criterion]:
    criteria: List[Criterion] = []
    for feature_name in feature_order:
        if feature_name not in criteria_defs:
            raise ValueError(f"Missing criterion definition for '{feature_name}'.")

        spec = criteria_defs[feature_name]
        crit_type = spec.get("type")
        if crit_type == "cardinal":
            shape = spec.get("shape", "gain")
            n_segments = int(spec.get("n_segments", 2))
            raw_min = spec.get("min", spec.get("min_val"))
            raw_max = spec.get("max", spec.get("max_val"))

            try:
                min_val = None if raw_min in (None, "") else float(raw_min)
                max_val = None if raw_max in (None, "") else float(raw_max)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cardinal criterion '{feature_name}' must have numeric min/max bounds when provided."
                ) from exc

            if min_val is not None and max_val is not None and min_val >= max_val:
                raise ValueError(
                    f"Cardinal criterion '{feature_name}' must satisfy min < max when both bounds are provided."
                )

            criteria.append(
                CardinalCriterion(
                    feature_name,
                    n_segments=n_segments,
                    shape=shape,
                    min_val=min_val,
                    max_val=max_val,
                )
            )
        elif crit_type == "ordinal":
            categories = spec.get("categories", [])
            if not categories:
                raise ValueError(f"Ordinal criterion '{feature_name}' requires categories.")
            criteria.append(OrdinalCriterion(feature_name, categories=categories))
        elif crit_type == "nominal":
            categories = spec.get("categories", [])
            if not categories:
                raise ValueError(f"Nominal criterion '{feature_name}' requires categories.")
            criteria.append(NominalCriterion(feature_name, categories=categories))
        else:
            raise ValueError(f"Invalid criterion type for '{feature_name}': {crit_type}")
    return criteria


def _apply_missing_value_treatment(
    alternatives_df: pd.DataFrame,
    criteria_defs: Dict[str, dict],
    treatment: str,
) -> pd.DataFrame:
    if treatment not in {"assumeAverageValue", "assumeZeroValue"}:
        return alternatives_df

    df = alternatives_df.copy()

    for criterion_name, spec in criteria_defs.items():
        if criterion_name not in df.columns:
            continue

        crit_type = spec.get("type")

        if crit_type == "cardinal":
            numeric = pd.to_numeric(df[criterion_name], errors="coerce")
            if treatment == "assumeZeroValue":
                fill_value = 0.0
            else:
                valid = numeric.dropna()
                fill_value = float(valid.mean()) if not valid.empty else 0.0
            df[criterion_name] = numeric.fillna(fill_value)
            continue

        if crit_type != "nominal":
            continue

        categories = spec.get("categories") or []
        as_text = df[criterion_name].astype("string")
        missing_mask = df[criterion_name].isna() | as_text.str.strip().eq("")
        if categories:
            missing_mask = missing_mask | (~as_text.isin(categories))

        if not bool(missing_mask.any()):
            continue

        if treatment == "assumeZeroValue":
            fill_category = str(categories[0]) if categories else ""
        else:
            valid_values = as_text[~missing_mask]
            if not valid_values.empty:
                fill_category = str(valid_values.mode().iloc[0])
            else:
                fill_category = str(categories[0]) if categories else ""

        df.loc[missing_mask, criterion_name] = fill_category

    return df


def _validate_dense_rankings(rankings: np.ndarray) -> None:
    if rankings.size == 0:
        raise ValueError("Rankings cannot be empty.")

    if np.isnan(rankings.astype(float)).any():
        raise ValueError("Rankings must not contain missing values.")

    rounded = np.rint(rankings).astype(int)
    if not np.allclose(rankings.astype(float), rounded.astype(float), atol=1e-9):
        raise ValueError("Rankings must be integer values.")

    unique = sorted(np.unique(rounded).tolist())
    expected = list(range(1, len(unique) + 1))
    if unique != expected:
        raise ValueError(
            f"Rankings must be dense and start at 1 (expected unique ranks {expected}, got {unique})."
        )


def run_uta_analysis(
    alternatives_df: pd.DataFrame,
    rankings: List[int],
    criteria_defs: Dict[str, dict],
    settings: dict,
    reference_names: Optional[List[str]] = None,
):
    _validate_reserved_criterion_names(criteria_defs)

    alternatives_processed = _apply_missing_value_treatment(
        alternatives_df=alternatives_df,
        criteria_defs=criteria_defs,
        treatment=str(settings.get("missing_value_treatment", "assumeAverageValue")),
    )

    validate_alternatives_schema(alternatives_processed, criteria_defs)
    missing_values = get_missing_required_values(alternatives_processed, criteria_defs)
    if missing_values:
        details = ", ".join([f"{name}: {count}" for name, count in missing_values.items()])
        raise ValueError(f"Cannot run analysis with missing criterion values ({details}).")

    X = alternatives_processed.set_index("Name")
    y_in = np.array(rankings, dtype=float)

    if reference_names:
        name_series = alternatives_processed["Name"].astype(str)
        reference_mask = name_series.isin(reference_names).to_numpy()
    else:
        reference_mask = np.ones(len(alternatives_processed), dtype=bool)

    if int(reference_mask.sum()) < 2:
        raise ValueError("Please select at least 2 reference alternatives.")

    n_all = len(alternatives_processed)
    n_ref = int(reference_mask.sum())

    if len(y_in) == n_all:
        y_selected = y_in[reference_mask]
        y_all_display = y_in.copy()
    elif len(y_in) == n_ref:
        y_selected = y_in.copy()
        y_all_display = np.full(n_all, np.nan)
        y_all_display[reference_mask] = y_in
    else:
        raise ValueError(
            "Rankings length must match either all alternatives or selected reference alternatives."
        )

    _validate_dense_rankings(y_selected)
    y_ref = np.rint(y_selected).astype(int)

    criteria = criteria_defs_to_objects(
        criteria_defs=criteria_defs,
        feature_order=list(X.columns),
    )

    model = UTAEstimator(
        criteria=criteria,
        algorithm=settings["algorithm"],
        sigma=settings["sigma"],
        breakpoints=str(settings.get("breakpoints", "quantile")),
        theta=settings.get("theta", 1.0),
        big_m=float(settings.get("big_m", 1000.0)),
        epsilon_sign=float(settings.get("ineq", 0.001)),
        max_nonmonotonicity_degree=int(settings.get("max_nonmonotonicity_degree", 2)),
        objective_threshold=float(settings.get("objective_threshold", 0.01)),
        minimum_improvement=float(settings.get("minimum_improvement", 0.0)),
    )

    fit_params = inspect.signature(model.fit).parameters
    if "reference_mask" in fit_params:
        model.fit(X, y_ref, reference_mask=reference_mask)
    else:
        X_ref = X.iloc[reference_mask]
        model.fit(X_ref, y_ref)

    utilities = model.predict(X)
    kendall_tau = float(model.score(X.iloc[reference_mask], y_ref))

    utilities_df = pd.DataFrame(
        {
            "Alternative": X.index,
            "Utility": utilities,
            "Actual Rank": y_all_display,
            "Reference": reference_mask,
        }
    )

    return {
        "model": model,
        "utilities_df": utilities_df,
        "objective_value": model.objective_value_,
        "kendall_tau": kendall_tau,
        "marginal_utilities": model.marginal_utilities_,
        "breakpoint_utilities": getattr(model, "breakpoint_utilities_", None),
        "partial_values": getattr(model, "partial_values_", None),
        "criteria": model.criteria_,
    }


def _coerce_reference_rank(raw_rank: Any, alt_name: str) -> int:
    if isinstance(raw_rank, bool):
        raise ValueError(f"Reference alternative '{alt_name}' has invalid rank {raw_rank!r} (bool is not allowed).")

    if isinstance(raw_rank, (int, np.integer)):
        return int(raw_rank)

    if isinstance(raw_rank, (float, np.floating)):
        if not np.isfinite(raw_rank):
            raise ValueError(f"Reference alternative '{alt_name}' has non-finite rank {raw_rank!r}.")
        rounded = int(round(float(raw_rank)))
        if not np.isclose(float(raw_rank), float(rounded), atol=1e-9):
            raise ValueError(f"Reference alternative '{alt_name}' has non-integer rank {raw_rank!r}.")
        return rounded

    raise ValueError(f"Reference alternative '{alt_name}' has invalid rank type: {type(raw_rank).__name__}.")


def _parse_alternatives_payload(alternatives_payload: Any) -> Tuple[pd.DataFrame, List[str], List[int]]:
    if isinstance(alternatives_payload, dict):
        required_keys = {"reference_alternatives", "non_reference_alternatives"}
        missing_keys = sorted(required_keys - set(alternatives_payload.keys()))
        if missing_keys:
            raise ValueError(
                "In JSON format, alternatives must include keys: "
                "`reference_alternatives` and `non_reference_alternatives`."
            )

        ref_rows = alternatives_payload.get("reference_alternatives")
        non_ref_rows = alternatives_payload.get("non_reference_alternatives")
        if not isinstance(ref_rows, list) or not isinstance(non_ref_rows, list):
            raise ValueError(
                "In JSON format, alternatives.reference_alternatives and "
                "alternatives.non_reference_alternatives must be arrays."
            )

        seen = set()
        merged_records = []
        reference_names: List[str] = []
        rankings: List[int] = []

        for idx, raw_row in enumerate(ref_rows):
            if not isinstance(raw_row, dict):
                raise ValueError(f"Reference alternative at index {idx} must be an object.")
            if "rank" not in raw_row:
                raise ValueError(
                    "Each object in alternatives.reference_alternatives must include integer `rank`."
                )

            row = dict(raw_row)
            raw_name = row.get("Name", row.get("name"))
            if raw_name in (None, ""):
                raise ValueError(f"Reference alternative at index {idx} is missing `Name`.")
            name = str(raw_name)
            if name in seen:
                raise ValueError(f"Duplicate alternative name '{name}' in JSON payload.")
            seen.add(name)
            rank_value = _coerce_reference_rank(row.pop("rank"), name)
            reference_names.append(name)
            rankings.append(rank_value)
            row["Name"] = name
            merged_records.append(row)

        for idx, raw_row in enumerate(non_ref_rows):
            if not isinstance(raw_row, dict):
                raise ValueError(f"Non-reference alternative at index {idx} must be an object.")

            row = dict(raw_row)
            if "rank" in row:
                raise ValueError(
                    "alternatives.non_reference_alternatives must not contain `rank`. "
                    "Only reference alternatives carry ranks."
                )

            raw_name = row.get("Name", row.get("name"))
            if raw_name in (None, ""):
                raise ValueError(f"Non-reference alternative at index {idx} is missing `Name`.")
            name = str(raw_name)
            if name in seen:
                raise ValueError(f"Duplicate alternative name '{name}' in JSON payload.")
            seen.add(name)
            row["Name"] = name
            merged_records.append(row)

        if not reference_names:
            raise ValueError(
                "alternatives.reference_alternatives must contain at least one alternative with `rank`."
            )

        merged_df = _normalize_name_column(pd.DataFrame(merged_records))
        _validate_dense_rankings(np.asarray(rankings, dtype=float))
        return merged_df, reference_names, rankings

    raise ValueError(
        "JSON import supports only this format: "
        "alternatives must be an object with reference_alternatives and non_reference_alternatives arrays."
    )


def load_project_from_json(
    uploaded_file,
) -> Tuple[str, str, Dict[str, dict], pd.DataFrame, List[int], List[str], dict | None, dict]:
    content = uploaded_file.read()
    data = json.loads(content.decode("utf-8"))

    for key in REQUIRED_JSON_KEYS:
        if key not in data:
            raise ValueError(f"JSON v2 missing required key: {key}")

    if "rankings" in data:
        raise ValueError(
            "Top-level `rankings` is no longer supported. "
            "Put `rank` inside each alternatives.reference_alternatives item."
        )

    project_name = data["project_name"]
    raw_description = data.get("description", "")
    project_description = "" if raw_description is None else str(raw_description)
    criteria_defs = data["criteria"]
    _validate_reserved_criterion_names(criteria_defs)
    alternatives_df, parsed_reference_names, rankings = _parse_alternatives_payload(data["alternatives"])
    loaded_results = _deserialize_results(data.get("results"))
    raw_settings = data.get("algorithm_settings")
    if isinstance(raw_settings, dict):
        forbidden = [field for field in ("reference_names", "output_folder") if field in raw_settings]
        if forbidden:
            joined = ", ".join(f"`{field}`" for field in forbidden)
            raise ValueError(
                f"Legacy algorithm_settings fields are not supported in strict JSON mode: {joined}."
            )
    algorithm_settings = _normalize_algorithm_settings(raw_settings)

    return (
        project_name,
        project_description,
        criteria_defs,
        alternatives_df,
        rankings,
        parsed_reference_names,
        loaded_results,
        algorithm_settings,
    )


def load_project_from_csv(uploaded_file) -> Tuple[str, Dict[str, dict], pd.DataFrame]:
    content = uploaded_file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    df = _normalize_name_column(df)

    criteria_defs: Dict[str, dict] = {}
    for col in df.columns:
        if col == "Name":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            criteria_defs[col] = {"type": "cardinal", "shape": "gain", "n_segments": 2}
        else:
            unique_vals = [str(v) for v in pd.Series(df[col]).dropna().unique().tolist()]
            criteria_defs[col] = {
                "type": None,
                "categories": unique_vals,
            }

    _validate_reserved_criterion_names(criteria_defs)

    project_name = uploaded_file.name.replace(".csv", "")
    return project_name, criteria_defs, df


def load_example_dataset(example_name: str):
    if example_name == "apartments":
        return load_apartments()
    if example_name == "cars":
        return load_cars()
    raise ValueError(f"Unknown example dataset: {example_name}")


def export_project_json(
    project_name: str,
    project_description: Optional[str],
    criteria_defs: Dict[str, dict],
    alternatives_df: pd.DataFrame,
    rankings,
    results,
    algorithm_settings: Optional[dict] = None,
    reference_names: Optional[List[str]] = None,
):
    _validate_reserved_criterion_names(criteria_defs)

    alt_records = alternatives_df.to_dict("records") if alternatives_df is not None else []
    all_names = [str(row.get("Name")) for row in alt_records]
    all_name_set = set(all_names)
    if len(set(all_names)) != len(all_names):
        raise ValueError("Cannot export JSON with duplicate alternative names.")

    if reference_names is None:
        reference_names = []
    else:
        reference_names = [str(name) for name in reference_names]

    if (
        not reference_names
        and results
        and results.get("utilities_df") is not None
        and "Reference" in results["utilities_df"].columns
    ):
        reference_names_from_results = (
            results["utilities_df"]
            .loc[results["utilities_df"]["Reference"] == True, "Alternative"]
            .astype(str)
            .tolist()
        )
        if reference_names_from_results:
            reference_names = reference_names_from_results

    if reference_names:
        if len(set(reference_names)) != len(reference_names):
            raise ValueError("Cannot export JSON with duplicate names in reference alternatives.")
        missing_refs = [name for name in reference_names if name not in all_name_set]
        if missing_refs:
            raise ValueError(
                "Cannot export JSON because some reference alternatives are missing from alternatives data: "
                + ", ".join(missing_refs)
            )

    rankings_list = [] if rankings is None else list(rankings)
    normalized_ranks: List[int] = []
    if reference_names:
        if len(rankings_list) == len(reference_names):
            normalized_ranks = [int(np.rint(float(v))) for v in rankings_list]
        elif len(rankings_list) == len(all_names):
            by_name = {name: rankings_list[idx] for idx, name in enumerate(all_names)}
            normalized_ranks = [int(np.rint(float(by_name[name]))) for name in reference_names]
        else:
            raise ValueError(
                "Cannot export JSON: rankings length must match either reference alternatives or all alternatives."
            )
        _validate_dense_rankings(np.asarray(normalized_ranks, dtype=float))

    name_to_record = {str(row.get("Name")): dict(row) for row in alt_records}
    ref_set = set(reference_names)
    ref_records = []
    for name, rank in zip(reference_names, normalized_ranks):
        row = dict(name_to_record[name])
        row["rank"] = int(rank)
        ref_records.append(row)
    non_ref_records = [dict(row) for row in alt_records if str(row.get("Name")) not in ref_set]

    payload = {
        "project_name": project_name,
        "description": "" if project_description is None else str(project_description),
        "criteria": criteria_defs,
        "algorithm_settings": _serialize_algorithm_settings_for_json(algorithm_settings),
        "alternatives": {
            "reference_alternatives": ref_records,
            "non_reference_alternatives": non_ref_records,
        },
    }
    if results:
        payload["results"] = {
            "objective_value": results["objective_value"],
            "kendall_tau": results["kendall_tau"],
            "marginal_utilities": results["marginal_utilities"],
            "utilities_df": (
                results["utilities_df"].to_dict("records")
                if results.get("utilities_df") is not None
                else None
            ),
        }
        if results.get("breakpoint_utilities") is not None:
            payload["results"]["breakpoint_utilities"] = {
                str(name): float(value)
                for name, value in results["breakpoint_utilities"].items()
            }
        if results.get("partial_values") is not None:
            payload["results"]["partial_values"] = {
                str(name): [float(value) for value in values]
                for name, values in results["partial_values"].items()
            }
    return json.dumps(payload, indent=2)
