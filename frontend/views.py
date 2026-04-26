"""Step renderers for Streamlit frontend."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
try:
    from streamlit_sortables import sort_items
except ImportError:  # fallback when dependency is not installed yet
    sort_items = None

from state import reset_state
from services import (
    expected_alternative_columns,
    export_project_json,
    get_missing_required_values,
    load_example_dataset,
    load_project_from_csv,
    load_project_from_json,
    run_uta_analysis,
    validate_alternatives_schema,
)


def _get_criterion_details(spec: dict) -> str:
    crit_type = spec.get("type")
    if crit_type == "cardinal":
        min_bound = spec.get("min", spec.get("min_val"))
        max_bound = spec.get("max", spec.get("max_val"))
        return (
            f"Shape: {spec.get('shape', 'gain')}, Segments: {spec.get('n_segments', 2)}, "
            f"Min: {'auto' if min_bound in (None, '') else min_bound}, "
            f"Max: {'auto' if max_bound in (None, '') else max_bound}"
        )
    if crit_type in {"ordinal", "nominal"}:
        return f"Categories: {len(spec.get('categories', []))}"
    return "Type not selected"


def _parse_optional_bound(raw_value: str, label: str) -> float | None:
    value = raw_value.strip()
    if not value:
        return None

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid number.") from exc


def _is_reserved_criterion_name(name: str) -> bool:
    return str(name).strip().lower() == "rank"


def _get_breakpoint_utilities_for_view(
    criterion,
    criterion_values,
    breakpoint_utilities: dict | None,
) -> list[float] | None:
    breakpoint_count = len(criterion.breakpoints)

    if isinstance(criterion_values, dict):
        values: list[float] = []
        for bp in criterion.breakpoints:
            var_name = bp.get_marginal_utility_var_name()
            candidates = (bp.index, str(bp.index), var_name, str(bp.position))
            for key in candidates:
                if key in criterion_values:
                    values.append(float(criterion_values[key]))
                    break
            else:
                values = []
                break
        if len(values) == breakpoint_count:
            return values

    if criterion_values is not None and not isinstance(criterion_values, (str, bytes, dict)):
        try:
            sequence = list(criterion_values)
        except TypeError:
            sequence = None
        if sequence is not None and len(sequence) >= breakpoint_count:
            return [float(sequence[idx]) for idx in range(breakpoint_count)]

    if isinstance(breakpoint_utilities, dict):
        values = []
        for bp in criterion.breakpoints:
            var_name = bp.get_marginal_utility_var_name()
            if var_name not in breakpoint_utilities:
                values = []
                break
            values.append(float(breakpoint_utilities[var_name]))
        if len(values) == breakpoint_count:
            return values

    return None


def _sync_alternatives_with_criteria() -> None:
    expected_columns = expected_alternative_columns(st.session_state.criteria_defs)
    alternatives_df = st.session_state.alternatives_df

    if alternatives_df is None:
        st.session_state.alternatives_df = pd.DataFrame(columns=expected_columns)
        return

    if "Name" not in alternatives_df.columns:
        raise ValueError("Alternatives table must contain a 'Name' column.")

    updated_df = alternatives_df.copy()
    for column in expected_columns:
        if column not in updated_df.columns:
            updated_df[column] = pd.NA

    ordered_existing = [col for col in expected_columns if col in updated_df.columns]
    remaining = [col for col in updated_df.columns if col not in ordered_existing]
    st.session_state.alternatives_df = updated_df[ordered_existing + remaining]


def _build_groups_from_rankings(
    reference_names: list[str] | None,
    rankings: list[int] | None,
    labels: list[str],
) -> list[list[str]] | None:
    if not reference_names or rankings is None or len(rankings) != len(reference_names):
        return None

    try:
        ranks_int = [int(value) for value in rankings]
    except Exception:
        return None

    unique_ranks = sorted(set(ranks_int))
    expected = list(range(1, len(unique_ranks) + 1))
    if unique_ranks != expected:
        return None

    rank_by_name = {str(name): rank for name, rank in zip(reference_names, ranks_int)}
    groups = [[name for name in labels if rank_by_name.get(name) == rank] for rank in unique_ranks]
    return groups


def render_home():
    st.markdown(
        """
    ## What is UTA?

    **UTA (UTilites Additives)** is a multi-criteria decision analysis method that learns
    your preferences from rankings. It discovers utility functions that best explain how
    you evaluate and compare alternatives.
    """
    )

    st.info(
        """
    **How It Works**

    1. Define Criteria - Specify what factors matter (price, quality, etc.)
    2. Add Alternatives - Input the options you are considering
    3. Rank Preferences - Order them from best to worst
    4. Learn Utilities - UTA discovers your hidden utility function
    5. Predict and Analyze - Use the learned model for new decisions
    """
    )

    st.markdown("---")
    st.markdown("## Get Started")

    if st.button("Start New Project", use_container_width=True, type="primary"):
            st.session_state.step = 1
            st.rerun()

    st.markdown("---")
    st.markdown("## Examples")
    ex1, ex2 = st.columns(2)
    with ex1:
        if st.button("Try Apartments Example", use_container_width=True):
            _load_example("apartments")
    with ex2:
        if st.button("Try Cars Example", use_container_width=True):
            _load_example("cars")


def render_project_setup():
    project_name = st.text_input(
        "Project Name",
        value=st.session_state.project_name or "",
        placeholder="e.g., Apartment Selection 2026",
    )
    project_description = st.text_area(
        "Description (optional)",
        value=st.session_state.project_description or "",
        placeholder="Describe your decision problem...",
    )

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload JSON or CSV file",
        type=["json", "csv"],
        help="Upload a full project JSON (split format) or an alternatives CSV table.",
    )
    st.caption(
        "JSON loads criteria, alternatives (with embedded reference ranks), and optional saved results. "
        "CSV loads alternatives only; criterion types for text columns are resolved in Step 3."
    )
    with st.expander("File format guide"):
        st.markdown(
            """
            **JSON (`.json`)**
            - Required keys: `project_name`, `criteria`, `alternatives`
            - Optional key: `description`
            - `alternatives` must contain:
              - `reference_alternatives` (array)
              - `non_reference_alternatives` (array)
            - Each item in `reference_alternatives` must include integer `rank`
            - `rankings` at top level is not supported
            - Optional: `algorithm_settings`, `results`
            - `algorithm_settings.breakpoints` can be `"quantile"` or `"uniform"`
            - If `results` are included, the app can jump directly to Results.

            **CSV (`.csv`)**
            - First column should be alternative name (`Name` or `name` is accepted)
            - Other columns are treated as criteria
            - Numeric columns are auto-marked as cardinal
            - Text columns are loaded as pending and must be set to ordinal/nominal in Step 3
            """
        )
        st.markdown("**Download sample files**")
        sample_dir = Path(__file__).resolve().parent / "sample_inputs"
        sample_files = sorted(
            [
                path
                for path in sample_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".json", ".csv"}
            ],
            key=lambda p: p.name.lower(),
        ) if sample_dir.exists() else []

        if sample_files:
            for sample_path in sample_files:
                mime = "application/json" if sample_path.suffix.lower() == ".json" else "text/csv"
                st.download_button(
                    label=f"Download {sample_path.name}",
                    data=sample_path.read_bytes(),
                    file_name=sample_path.name,
                    mime=mime,
                    key=f"download_sample_{sample_path.name}",
                )
        else:
            st.caption("No sample files found in `frontend/sample_inputs`.")
    if uploaded_file is not None:
        if st.button("Load Uploaded File", use_container_width=True):
            try:
                if uploaded_file.name.endswith(".json"):
                    (
                        p_name,
                        p_description,
                        criteria_defs,
                        alternatives_df,
                        rankings,
                        reference_names,
                        loaded_results,
                        loaded_algorithm_settings,
                    ) = load_project_from_json(uploaded_file)
                    validate_alternatives_schema(alternatives_df, criteria_defs)
                    st.session_state.project_name = p_name
                    st.session_state.project_description = p_description
                    st.session_state.criteria_defs = criteria_defs
                    st.session_state.alternatives_df = alternatives_df
                    st.session_state.rankings = rankings
                    st.session_state.reference_names = reference_names
                    st.session_state.ranking_groups = None
                    st.session_state.algorithm_settings = {
                        **st.session_state.algorithm_settings,
                        **loaded_algorithm_settings,
                    }
                    st.session_state.results = loaded_results
                    st.session_state.model = None
                    if loaded_results is not None:
                        st.session_state.step = 7
                    else:
                        st.session_state.step = 6 if rankings else 4
                    st.success(f"Loaded project: {p_name}")
                    st.rerun()
                elif uploaded_file.name.endswith(".csv"):
                    p_name, criteria_defs, alternatives_df = load_project_from_csv(uploaded_file)
                    validate_alternatives_schema(alternatives_df, criteria_defs)
                    st.session_state.project_name = st.session_state.project_name or f"Project: {p_name}"
                    st.session_state.criteria_defs = criteria_defs
                    st.session_state.alternatives_df = alternatives_df
                    st.session_state.reference_names = None
                    st.session_state.ranking_groups = None
                    st.session_state.step = 3
                    st.success(f"Loaded {len(alternatives_df)} alternatives from CSV")
                    st.info("Please finish criterion type selection in Step 3.")
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {e}")

    st.markdown("---")
    st.markdown("### Or Load Example Dataset")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Apartments", use_container_width=True):
            _load_example("apartments")
    with c2:
        if st.button("Cars", use_container_width=True):
            _load_example("cars")

    st.markdown("---")
    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        if st.button("Back to Home", use_container_width=True):
            st.session_state.step = 0
            st.rerun()
    with b2:
        if st.session_state.results is not None:
            if st.button("Jump to Results", use_container_width=True):
                st.session_state.step = 7
                st.rerun()
    with b3:
        if st.button("Next: Algorithm Settings", use_container_width=True, type="primary"):
            if project_name:
                st.session_state.project_name = project_name
                st.session_state.project_description = project_description
                st.session_state.step = 2
                st.rerun()
            else:
                st.error("Please enter a project name")


def render_define_criteria():
    st.markdown(f"**Project:** {st.session_state.project_name}")

    if st.session_state.criteria_defs:
        st.markdown("### Current Criteria")
        criteria_df = pd.DataFrame(
            [
                {
                    "Criterion": name,
                    "Type": spec.get("type") or "pending",
                    "Details": _get_criterion_details(spec),
                }
                for name, spec in st.session_state.criteria_defs.items()
            ]
        )
        st.dataframe(criteria_df, use_container_width=True)

        with st.expander("Open Modify / Remove Criteria", expanded=False):
            if any(spec.get("type") is None for spec in st.session_state.criteria_defs.values()):
                st.warning("Some criteria are pending type selection. Resolve them in their expanders below.")
            for name, spec in list(st.session_state.criteria_defs.items()):
                criterion_type = spec.get("type")
                with st.expander(f"{name} ({criterion_type or 'pending'})"):
                    if criterion_type == "cardinal":
                        shape_value = st.selectbox(
                            "Shape",
                            options=["gain", "cost"],
                            index=0 if spec.get("shape", "gain") == "gain" else 1,
                            key=f"edit_shape_{name}",
                        )
                        segments_value = int(
                            st.number_input(
                                "Segments",
                                min_value=1,
                                max_value=20,
                                value=int(spec.get("n_segments", 2)),
                                step=1,
                                key=f"edit_segments_{name}",
                            )
                        )
                        col_min, col_max = st.columns(2)
                        with col_min:
                            min_text = st.text_input(
                                "Min (optional)",
                                value="" if spec.get("min", spec.get("min_val")) in (None, "") else str(spec.get("min", spec.get("min_val"))),
                                key=f"edit_min_{name}",
                            )
                        with col_max:
                            max_text = st.text_input(
                                "Max (optional)",
                                value="" if spec.get("max", spec.get("max_val")) in (None, "") else str(spec.get("max", spec.get("max_val"))),
                                key=f"edit_max_{name}",
                            )

                        action_col1, action_col2 = st.columns(2)
                        with action_col1:
                            if st.button(f"Save changes for {name}", key=f"save_criterion_{name}", use_container_width=True):
                                if _is_reserved_criterion_name(name):
                                    st.error("Criterion name `rank` is reserved. Please remove it and use a different name.")
                                    continue
                                try:
                                    min_bound = _parse_optional_bound(min_text, "Min")
                                    max_bound = _parse_optional_bound(max_text, "Max")
                                except ValueError as exc:
                                    st.error(str(exc))
                                    continue

                                if min_bound is not None and max_bound is not None and min_bound >= max_bound:
                                    st.error("Min must be smaller than max.")
                                    continue

                                updated_spec = {
                                    "type": "cardinal",
                                    "shape": shape_value,
                                    "n_segments": segments_value,
                                }
                                if min_bound is not None:
                                    updated_spec["min"] = min_bound
                                if max_bound is not None:
                                    updated_spec["max"] = max_bound
                                st.session_state.criteria_defs[name] = updated_spec
                                _sync_alternatives_with_criteria()
                                st.success(f"Updated criterion: {name}")
                                st.rerun()
                        with action_col2:
                            if st.button(f"Remove {name}", key=f"remove_criterion_{name}", use_container_width=True):
                                st.session_state.criteria_defs.pop(name, None)
                                if st.session_state.alternatives_df is not None and name in st.session_state.alternatives_df.columns:
                                    st.session_state.alternatives_df = st.session_state.alternatives_df.drop(columns=[name])
                                _sync_alternatives_with_criteria()
                                st.success(f"Removed criterion: {name}")
                                st.rerun()

                    elif criterion_type in {"ordinal", "nominal"}:
                        categories_text = st.text_area(
                            "Categories (one per line)",
                            value="\n".join(spec.get("categories", [])),
                            key=f"edit_categories_{name}",
                        )

                        action_col1, action_col2 = st.columns(2)
                        with action_col1:
                            if st.button(f"Save changes for {name}", key=f"save_criterion_{name}", use_container_width=True):
                                if _is_reserved_criterion_name(name):
                                    st.error("Criterion name `rank` is reserved. Please remove it and use a different name.")
                                    continue
                                categories = [c.strip() for c in categories_text.split("\n") if c.strip()]
                                if not categories:
                                    st.error("Please enter at least one category.")
                                    continue
                                st.session_state.criteria_defs[name] = {
                                    "type": criterion_type,
                                    "categories": categories,
                                }
                                _sync_alternatives_with_criteria()
                                st.success(f"Updated criterion: {name}")
                                st.rerun()
                        with action_col2:
                            if st.button(f"Remove {name}", key=f"remove_criterion_{name}", use_container_width=True):
                                st.session_state.criteria_defs.pop(name, None)
                                if st.session_state.alternatives_df is not None and name in st.session_state.alternatives_df.columns:
                                    st.session_state.alternatives_df = st.session_state.alternatives_df.drop(columns=[name])
                                _sync_alternatives_with_criteria()
                                st.success(f"Removed criterion: {name}")
                                st.rerun()

                    elif criterion_type is None:
                        selected_type = st.selectbox(
                            "Type",
                            options=["ordinal", "nominal"],
                            key=f"edit_pending_type_{name}",
                        )
                        categories_text = st.text_area(
                            "Categories (one per line)",
                            value="\n".join(spec.get("categories", [])),
                            key=f"edit_pending_categories_{name}",
                        )

                        action_col1, action_col2 = st.columns(2)
                        with action_col1:
                            if st.button(f"Save changes for {name}", key=f"save_criterion_{name}", use_container_width=True):
                                if _is_reserved_criterion_name(name):
                                    st.error("Criterion name `rank` is reserved. Please remove it and use a different name.")
                                    continue
                                categories = [c.strip() for c in categories_text.split("\n") if c.strip()]
                                if not categories:
                                    st.error("Please enter at least one category.")
                                    continue
                                st.session_state.criteria_defs[name] = {
                                    "type": selected_type,
                                    "categories": categories,
                                }
                                _sync_alternatives_with_criteria()
                                st.success(f"Updated criterion: {name}")
                                st.rerun()
                        with action_col2:
                            if st.button(f"Remove {name}", key=f"remove_criterion_{name}", use_container_width=True):
                                st.session_state.criteria_defs.pop(name, None)
                                if st.session_state.alternatives_df is not None and name in st.session_state.alternatives_df.columns:
                                    st.session_state.alternatives_df = st.session_state.alternatives_df.drop(columns=[name])
                                _sync_alternatives_with_criteria()
                                st.success(f"Removed criterion: {name}")
                                st.rerun()

    with st.expander("Add Criteria", expanded=False):
        col1, col2 = st.columns([2, 1])

        with col1:
            criterion_name = st.text_input("Criterion Name", placeholder="e.g., Price")
            criterion_type = st.selectbox(
                "Criterion Type",
                options=["cardinal", "ordinal", "nominal"],
            )
            shape = "gain"
            cardinal_segments = 2
            cardinal_min = ""
            cardinal_max = ""
            categories = ""
            if criterion_type == "cardinal":
                shape = st.selectbox("Cardinal Shape", options=["gain", "cost"])
                cardinal_segments = st.number_input(
                    "Cardinal Segments",
                    min_value=1,
                    max_value=20,
                    value=2,
                    step=1,
                )
                min_col, max_col = st.columns(2)
                with min_col:
                    cardinal_min = st.text_input("Cardinal Min (optional)", placeholder="auto")
                with max_col:
                    cardinal_max = st.text_input("Cardinal Max (optional)", placeholder="auto")
            else:
                categories = st.text_area(
                    "Categories (one per line)",
                    placeholder="Poor\nFair\nGood\nExcellent",
                )

        with col2:
            st.info(
            """
            Cardinal: numeric values

            Ordinal: ordered categories

            Nominal: unordered categories
            """
            )

        if st.button("Add Criterion"):
            if not criterion_name:
                st.error("Please enter a criterion name")
            elif _is_reserved_criterion_name(criterion_name):
                st.error("Criterion name `rank` is reserved for reference ranking metadata. Choose a different name.")
            elif criterion_type == "cardinal":
                try:
                    min_bound = _parse_optional_bound(cardinal_min, "Min")
                    max_bound = _parse_optional_bound(cardinal_max, "Max")
                except ValueError as exc:
                    st.error(str(exc))
                    return

                if min_bound is not None and max_bound is not None and min_bound >= max_bound:
                    st.error("Min must be smaller than max.")
                    return

                st.session_state.criteria_defs[criterion_name] = {
                    "type": "cardinal",
                    "shape": shape,
                    "n_segments": int(cardinal_segments),
                }
                if min_bound is not None:
                    st.session_state.criteria_defs[criterion_name]["min"] = min_bound
                if max_bound is not None:
                    st.session_state.criteria_defs[criterion_name]["max"] = max_bound
                _sync_alternatives_with_criteria()
                st.success(f"Added criterion: {criterion_name}")
                st.rerun()
            else:
                cat_list = [c.strip() for c in categories.split("\n") if c.strip()]
                if not cat_list:
                    st.error("Please enter categories")
                else:
                    st.session_state.criteria_defs[criterion_name] = {
                        "type": criterion_type,
                        "categories": cat_list,
                    }
                    _sync_alternatives_with_criteria()
                    st.success(f"Added criterion: {criterion_name}")
                    st.rerun()

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Back", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with c2:
        if st.button("Next: Add Alternatives", use_container_width=True, type="primary"):
            reserved_names = [
                name for name in st.session_state.criteria_defs.keys()
                if _is_reserved_criterion_name(name)
            ]
            if reserved_names:
                st.error(
                    "Criterion name `rank` is reserved. "
                    f"Please remove/rename: {', '.join(reserved_names)}"
                )
                return
            unresolved = [n for n, spec in st.session_state.criteria_defs.items() if spec.get("type") is None]
            if unresolved:
                st.error(f"Please resolve criterion types first: {', '.join(unresolved)}")
            elif st.session_state.criteria_defs:
                st.session_state.step = 4
                st.rerun()
            else:
                st.error("Please add at least one criterion")
    with c3:
        if st.session_state.results is not None:
            if st.button("Jump to Results", use_container_width=True):
                st.session_state.step = 7
                st.rerun()


def render_add_alternatives():
    st.markdown(f"**Project:** {st.session_state.project_name}")

    _sync_alternatives_with_criteria()

    if not st.session_state.alternatives_df.empty:
        st.markdown("### Current Alternatives")
        st.dataframe(st.session_state.alternatives_df, use_container_width=True, hide_index=True)
    else:
        st.info("No alternatives yet. Add your first alternative below.")

    with st.expander("Add New Alternative", expanded=True):
        alt_name = st.text_input("Alternative Name", placeholder="e.g., Alternative A")
        criterion_values = {}
        cols = st.columns(max(1, min(3, len(st.session_state.criteria_defs))))

        for idx, (crit_name, spec) in enumerate(st.session_state.criteria_defs.items()):
            with cols[idx % len(cols)]:
                if spec.get("type") == "cardinal":
                    _nc, _cc = st.columns([4, 1])
                    with _cc:
                        st.write("&nbsp;", unsafe_allow_html=True)
                        is_na = st.checkbox("N/A", key=f"add_na_{crit_name}", value=bool(st.session_state.get(f"add_na_{crit_name}", False)))
                    with _nc:
                        num_val = st.number_input(crit_name, key=f"crit_{crit_name}", disabled=is_na)
                    criterion_values[crit_name] = None if is_na else num_val
                else:
                    categories = spec.get("categories") or []
                    is_nominal = spec.get("type") == "nominal"
                    options = (["(missing)"] + categories) if is_nominal else categories
                    selected = st.selectbox(crit_name, options=options, key=f"crit_{crit_name}")
                    criterion_values[crit_name] = None if selected == "(missing)" else selected

        if st.button("Add Alternative", use_container_width=True, key="add_alternative_button"):
            if not alt_name.strip():
                st.error("Please enter an alternative name.")
            else:
                new_row = {"Name": alt_name.strip(), **criterion_values}
                st.session_state.alternatives_df = pd.concat(
                    [st.session_state.alternatives_df, pd.DataFrame([new_row])], ignore_index=True
                )
                st.success(f"Added alternative: {alt_name.strip()}")
                st.rerun()

    if not st.session_state.alternatives_df.empty and st.session_state.criteria_defs:
        with st.expander("Fill or Update Alternative Values", expanded=True):
            selected_alt_idx = st.selectbox(
                "Select alternative",
                options=st.session_state.alternatives_df.index.tolist(),
                format_func=lambda idx: str(st.session_state.alternatives_df.iloc[idx]["Name"]),
                key="update_alternative_idx",
            )
            updated_values = {}
            cols = st.columns(max(1, min(3, len(st.session_state.criteria_defs))))
            selected_row = st.session_state.alternatives_df.iloc[selected_alt_idx]

            for idx, (crit_name, spec) in enumerate(st.session_state.criteria_defs.items()):
                with cols[idx % len(cols)]:
                    current_value = selected_row.get(crit_name)
                    if spec.get("type") == "cardinal":
                        default_numeric = pd.to_numeric(pd.Series([current_value]), errors="coerce").iloc[0]
                        current_is_na = pd.isna(default_numeric)
                        input_default = float(default_numeric) if pd.notna(default_numeric) else 0.0
                        na_key = f"update_na_{crit_name}_{selected_alt_idx}"
                        val_key = f"update_{crit_name}_{selected_alt_idx}"
                        if na_key not in st.session_state:
                            st.session_state[na_key] = bool(current_is_na)

                        _nc, _cc = st.columns([4, 1])
                        with _cc:
                            st.write("&nbsp;", unsafe_allow_html=True)
                            is_na = st.checkbox("N/A", key=na_key)
                        with _nc:
                            num_val = st.number_input(
                                f"{crit_name} (for {selected_row['Name']})",
                                value=input_default,
                                key=val_key,
                                disabled=is_na,
                            )
                        updated_values[crit_name] = None if is_na else num_val
                    else:
                        categories = spec.get("categories") or []
                        is_nominal = spec.get("type") == "nominal"
                        options = (["(missing)"] + categories) if is_nominal else categories
                        current_text = "" if pd.isna(current_value) else str(current_value)
                        default_index = options.index(current_text) if current_text in options else 0
                        selected = st.selectbox(
                            f"{crit_name} (for {selected_row['Name']})",
                            options=options,
                            index=default_index,
                            key=f"update_{crit_name}_{selected_alt_idx}",
                        ) if options else None
                        updated_values[crit_name] = None if (selected is None or selected == "(missing)") else selected

            if st.button("Save Values", use_container_width=True, key=f"save_values_button_{selected_alt_idx}"):
                for crit_name, value in updated_values.items():
                    st.session_state.alternatives_df.at[selected_alt_idx, crit_name] = (
                        pd.NA if value is None else value
                    )
                st.success(f"Updated values for {selected_row['Name']}")
                st.rerun()

    if not st.session_state.alternatives_df.empty:
        with st.expander("Remove Alternative", expanded=False):
            remove_options = st.session_state.alternatives_df.index.tolist()
            selected_remove_idx = st.selectbox(
                "Select alternative to remove",
                options=remove_options,
                format_func=lambda idx: str(st.session_state.alternatives_df.iloc[idx]["Name"]),
                key="remove_alternative_idx",
            )
            if st.button("Remove Selected Alternative", use_container_width=True):
                removed_name = str(st.session_state.alternatives_df.iloc[selected_remove_idx]["Name"])
                st.session_state.alternatives_df = (
                    st.session_state.alternatives_df.drop(index=selected_remove_idx).reset_index(drop=True)
                )
                if st.session_state.reference_names:
                    st.session_state.reference_names = [
                        name for name in st.session_state.reference_names if name != removed_name
                    ]
                st.session_state.rankings = None
                st.session_state.pop("rank_order_labels", None)
                st.session_state.pop("ranking_groups", None)
                st.success(f"Removed alternative: {removed_name}")
                st.rerun()

    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    missing_required_values = get_missing_required_values(
        st.session_state.alternatives_df,
        st.session_state.criteria_defs,
    )
    if missing_required_values:
        details = "\n".join([
            f"- **{criterion_name}**: {count} missing or invalid value(s)"
            for criterion_name, count in missing_required_values.items()
        ])
        st.info(
            "Some criterion values are missing and will be imputed automatically "
            "using the Missing Value Treatment selected in Algorithm Settings:\n"
            f"{details}"
        )

    with c1:
        if st.button("Back", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
    with c2:
        if st.button("Next: Rank Preferences", use_container_width=True, type="primary"):
            if st.session_state.alternatives_df is not None and len(st.session_state.alternatives_df) >= 2:
                st.session_state.step = 5
                st.rerun()
            else:
                st.error("Please add at least 2 alternatives")
    with c3:
        if st.session_state.results is not None:
            if st.button("Jump to Results", use_container_width=True):
                st.session_state.step = 7
                st.rerun()


def render_rank_preferences():
    st.markdown(f"**Project:** {st.session_state.project_name}")

    all_names = st.session_state.alternatives_df["Name"].astype(str).tolist()
    if len(set(all_names)) != len(all_names):
        st.error("Alternative names must be unique to use grouped ranking.")
        return

    saved_ref = st.session_state.reference_names or all_names
    saved_ref = [n for n in saved_ref if n in all_names] or all_names.copy()
    loaded_reference_order = saved_ref.copy()

    reference_names_selected = st.multiselect(
        "Reference alternatives (used for fitting)",
        options=all_names,
        default=saved_ref,
        help="Only these alternatives are ranked and used to learn preferences. Non-reference alternatives are scored but not used in fitting.",
    )
    if len(reference_names_selected) < 2:
        st.warning("Select at least 2 reference alternatives.")
        reference_names_selected = saved_ref

    st.session_state.reference_names = reference_names_selected

    non_reference = [n for n in all_names if n not in reference_names_selected]
    if non_reference:
        st.info(f"Non-reference (scored but not ranked): {', '.join(non_reference)}")

    alternatives = reference_names_selected

    st.info(
        "Create rank groups (big boxes). Alternatives inside the same group are treated as equal rank. "
        "Order groups from best to worst."
    )

    def _normalize_groups(raw_groups, labels, keep_empty: bool = False):
        seen = set()
        normalized = []
        for group in raw_groups:
            filtered = [name for name in group if name in labels and name not in seen]
            if filtered:
                normalized.append(filtered)
                seen.update(filtered)
            elif keep_empty:
                normalized.append([])
        for name in labels:
            if name not in seen:
                normalized.append([name])
                seen.add(name)
        return normalized if normalized else [[name] for name in labels]

    def _dense_rankings_from_groups(groups, labels):
        rank_by_name = {}
        for idx, group in enumerate(groups, start=1):
            for name in group:
                rank_by_name[name] = idx
        return [rank_by_name[name] for name in labels]

    if "ranking_groups" not in st.session_state:
        st.session_state.ranking_groups = None

    groups = st.session_state.ranking_groups
    if not groups or not isinstance(groups, list):
        groups = _build_groups_from_rankings(
            reference_names=loaded_reference_order,
            rankings=st.session_state.rankings,
            labels=alternatives,
        )
        if groups is None:
            groups = [[name] for name in alternatives]

    groups = _normalize_groups(groups, alternatives, keep_empty=True)

    control_col1, control_col2 = st.columns(2)
    with control_col1:
        if st.button("Add Rank Group", use_container_width=True):
            groups.append([])
            st.session_state.ranking_groups = groups
            st.rerun()
    with control_col2:
        if st.button("Remove Empty Groups", use_container_width=True):
            groups = [group for group in groups if len(group) > 0]
            if not groups:
                groups = [[name] for name in alternatives]
            st.session_state.ranking_groups = groups
            st.rerun()

    groups = _normalize_groups(groups, alternatives, keep_empty=True)

    st.markdown("### Rank Groups (best to worst)")
    custom_style = """

        .sortable-container,
        .sortable-container-body,
        .sortable-container-boy {
            background: transparent !important;
        }
        .sortable-item {
            background: rgba(151, 166, 195, 0.18) !important;
            color: var(--text-color) !important;
            border: 1px solid #ff4b4b !important;
            border-radius: 8px !important;
            padding: 8px 10px !important;
            margin: 6px 0 !important;
            min-height: 36px !important;
            line-height: 1.25 !important;
            font-size: 0.9rem !important;
            box-shadow: none !important;
        }
    """

    nested_drag_supported = False
    if sort_items is not None:
        containers = [
            {"header": f"Rank {idx + 1}", "items": list(group)}
            for idx, group in enumerate(groups)
        ]
        sorted_containers = None
        try:
            sorted_containers = sort_items(
                items=containers,
                multi_containers=True,
                direction="vertical",
                custom_style=custom_style,
                key=f"rank_nested_sortable_{len(groups)}",
            )
            nested_drag_supported = True
        except TypeError:
            nested_drag_supported = False

        if nested_drag_supported and sorted_containers:
            parsed_groups = []
            if isinstance(sorted_containers, list):
                for container in sorted_containers:
                    if isinstance(container, dict):
                        parsed_groups.append([str(name) for name in container.get("items", [])])
                    elif isinstance(container, list):
                        parsed_groups.append([str(name) for name in container])
            if parsed_groups and len(parsed_groups) == len(groups):
                groups = _normalize_groups(parsed_groups, alternatives, keep_empty=True)

        empty_group_indices = [idx + 1 for idx, group in enumerate(groups) if len(group) == 0]
        if empty_group_indices:
            labels = ", ".join(f"Rank {r}" for r in empty_group_indices)
            st.caption(f"Empty group(s): {labels} — drag alternatives here to fill them.")

    if not nested_drag_supported:
        st.warning(
            "Nested drag-and-drop is not available in the installed sortable component version. "
            "Using per-group drag ordering only."
        )
        group_labels = [f"Group {idx + 1}" for idx in range(len(groups))]
        if sort_items is not None and len(group_labels) > 1:
            ordered_labels = sort_items(
                items=group_labels,
                direction="vertical",
                custom_style=custom_style,
                key="rank_group_order_sortable_fallback",
            )
            if ordered_labels:
                label_to_group = {label: groups[idx] for idx, label in enumerate(group_labels)}
                groups = [label_to_group[label] for label in ordered_labels if label in label_to_group]

        for idx, group in enumerate(groups):
            st.markdown(f"**Rank {idx + 1}**")
            if sort_items is not None and len(group) > 1:
                ordered_group = sort_items(
                    items=group,
                    direction="vertical",
                    custom_style=custom_style,
                    key=f"rank_group_items_fallback_{idx}",
                )
                if ordered_group:
                    groups[idx] = [name for name in ordered_group if name in group]
                    for name in group:
                        if name not in groups[idx]:
                            groups[idx].append(name)
            if len(groups[idx]) == 0:
                st.caption("(empty group)")
            else:
                st.write(" · ".join(groups[idx]))

    groups_for_rankings = [group for group in groups if len(group) > 0]
    if not groups_for_rankings:
        groups_for_rankings = [[name] for name in alternatives]

    groups_for_rankings = _normalize_groups(groups_for_rankings, alternatives, keep_empty=False)
    rankings_dense = _dense_rankings_from_groups(groups_for_rankings, alternatives)

    st.session_state.ranking_groups = groups
    st.session_state.rankings = rankings_dense

    st.markdown("### Alternatives overview")
    overview_df = st.session_state.alternatives_df.copy()
    if st.session_state.rankings is not None and len(st.session_state.rankings) == len(alternatives):
        rank_map = {name: rank for name, rank in zip(alternatives, st.session_state.rankings)}
        overview_df["_is_reference"] = overview_df["Name"].astype(str).isin(alternatives)
        overview_df["_rank"] = overview_df["Name"].astype(str).map(rank_map)
        overview_df = overview_df.sort_values(
            by=["_is_reference", "_rank", "Name"],
            ascending=[False, True, True],
            kind="stable",
            na_position="last",
        ).drop(columns=["_is_reference", "_rank"])

    st.dataframe(overview_df, use_container_width=True, hide_index=True)

    has_empty_groups = any(len(g) == 0 for g in groups)
    if has_empty_groups:
        st.error("All rank groups must contain at least one alternative. Remove empty groups or drag alternatives into them before proceeding.")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Back", use_container_width=True):
            st.session_state.step = 4
            st.rerun()
    with c2:
        if st.button("Next: Summary", use_container_width=True, type="primary", disabled=has_empty_groups):
            st.session_state.step = 6
            st.rerun()
    with c3:
        if st.session_state.results is not None:
            if st.button("Jump to Results", use_container_width=True, disabled=has_empty_groups):
                st.session_state.step = 7
                st.rerun()


def render_algorithm_settings():
    st.markdown(f"**Project:** {st.session_state.project_name}")

    current_algorithm = str(st.session_state.algorithm_settings.get("algorithm", "UTASTAR")).upper()
    algorithm_options = ["UTASTAR", "UTANM"]
    algorithm_index = algorithm_options.index(current_algorithm) if current_algorithm in algorithm_options else 0

    saved_theta = float(st.session_state.algorithm_settings.get("theta", 1.0))
    saved_breakpoints = str(st.session_state.algorithm_settings.get("breakpoints", "quantile")).lower()
    saved_big_m = float(st.session_state.algorithm_settings.get("big_m", 1000.0))
    saved_ineq = float(st.session_state.algorithm_settings.get("ineq", 0.001))
    saved_max_nonmonotonicity_degree = int(st.session_state.algorithm_settings.get("max_nonmonotonicity_degree", 2))
    saved_objective_threshold = float(st.session_state.algorithm_settings.get("objective_threshold", 0.01))
    saved_minimum_improvement = float(st.session_state.algorithm_settings.get("minimum_improvement", 0.0))
    breakpoint_options = {
        "Quantile": "quantile",
        "Uniform": "uniform",
    }
    default_breakpoint_label = next(
        (label for label, value in breakpoint_options.items() if value == saved_breakpoints),
        "Quantile",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Task Settings")
        algorithm = st.selectbox(
            "Algorithm",
            options=algorithm_options,
            index=algorithm_index,
            help="Choose the optimization model used for fitting.",
        )
        sigma = st.number_input(
            "Sigma",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.algorithm_settings.get("sigma", 0.001)),
            format="%.4f",
            help="Minimum utility difference enforced for strict pairwise preferences.",
        )
        selected_breakpoint_label = st.selectbox(
            "Cardinal Breakpoint Mode",
            options=list(breakpoint_options.keys()),
            index=list(breakpoint_options.keys()).index(default_breakpoint_label),
            help=(
                "Quantile places cardinal breakpoints by data percentiles; "
                "Uniform spaces them evenly between the criterion min and max bounds."
            ),
        )
        breakpoints = breakpoint_options[selected_breakpoint_label]

        missing_value_treatment_labels = {
            "Assume Average Value": "assumeAverageValue",
            "Assume Zero Value": "assumeZeroValue",
        }
        missing_value_treatment_value = str(
            st.session_state.algorithm_settings.get("missing_value_treatment", "assumeAverageValue")
        )
        default_missing_label = next(
            (label for label, value in missing_value_treatment_labels.items() if value == missing_value_treatment_value),
            "Assume Average Value",
        )
        selected_missing_value_treatment_label = st.selectbox(
            "Missing Value Treatment For Cardinal And Nominal Criteria",
            options=list(missing_value_treatment_labels.keys()),
            index=list(missing_value_treatment_labels.keys()).index(default_missing_label),
            help=(
                "Defines how missing values are imputed before fitting: "
                "Assume Average Value uses mean/mode; Assume Zero Value uses 0 for cardinal and first category for nominal."
            ),
        )
        missing_value_treatment = missing_value_treatment_labels[selected_missing_value_treatment_label]

        theta = saved_theta
        big_m = saved_big_m
        ineq = saved_ineq
        max_nonmonotonicity_degree = saved_max_nonmonotonicity_degree
        objective_threshold = saved_objective_threshold
        minimum_improvement = saved_minimum_improvement

        if algorithm == "UTANM":
            theta = st.number_input(
                "Theta (shape-change penalty)",
                min_value=0.0,
                max_value=100.0,
                value=theta,
                format="%.4f",
                help="Penalty weight for each non-monotonic shape change in UTA-NM (higher = stronger preference for monotonicity).",
            )
            big_m = st.number_input(
                "BIGM",
                min_value=1.0,
                value=big_m,
                format="%.4f",
                help="Big-M constant used in UTA-NM mixed-integer constraints.",
            )
            ineq = st.number_input(
                "INEQ",
                min_value=0.0,
                max_value=1.0,
                value=ineq,
                format="%.4f",
                help="Inequality tolerance mapped to epsilon_sign.",
            )
            max_nonmonotonicity_degree = st.number_input(
                "Max Nonmonotonicity Degree",
                min_value=0,
                value=max_nonmonotonicity_degree,
                step=1,
                help="Maximum allowed number of nonmonotonic shape-change points in UTA-NM.",
            )
            objective_threshold = st.number_input(
                "Objective Threshold",
                min_value=0.0,
                value=objective_threshold,
                format="%.6f",
                help="Stop UTA-NM degree expansion once objective value is at or below this threshold.",
            )
            minimum_improvement = st.number_input(
                "Minimum Improvement Of Objective By Additional Degree",
                min_value=0.0,
                value=minimum_improvement,
                format="%.6f",
                help="Required objective decrease to accept a higher nonmonotonicity degree in UTA-NM.",
            )

    with col2:
        st.markdown("### Summary")
        st.write(f"**Criteria:** {len(st.session_state.criteria_defs)}")
        st.write(f"**Alternatives:** {len(st.session_state.alternatives_df) if st.session_state.alternatives_df is not None else 0}")
        st.write(f"**Algorithm:** {algorithm}")
        st.write(f"**Sigma:** {sigma:.4f}")
        st.write(f"**Cardinal breakpoints:** {breakpoints}")
        if algorithm == "UTANM":
            st.write(f"**BIGM:** {big_m:.4f}")
            st.write(f"**INEQ:** {ineq:.4f}")
            st.write(f"**Max nonmonotonicity degree:** {int(max_nonmonotonicity_degree)}")
            st.write(f"**Objective threshold:** {objective_threshold:.6f}")
            st.write(f"**Minimum improvement:** {minimum_improvement:.6f}")

    st.session_state.algorithm_settings = {
        "algorithm": algorithm,
        "sigma": sigma,
        "breakpoints": breakpoints,
        "theta": theta,
        "big_m": big_m,
        "ineq": ineq,
        "max_nonmonotonicity_degree": max_nonmonotonicity_degree,
        "objective_threshold": objective_threshold,
        "minimum_improvement": minimum_improvement,
        "missing_value_treatment": missing_value_treatment,
    }

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with c2:
        if st.button("Next: Define Criteria", use_container_width=True, type="primary"):
            st.session_state.step = 3
            st.rerun()
    with c3:
        if st.session_state.results is not None:
            if st.button("Jump to Results", use_container_width=True):
                st.session_state.step = 7
                st.rerun()


def render_results():
    st.markdown(f"**Project:** {st.session_state.project_name}")
    if st.session_state.project_description:
        st.caption(st.session_state.project_description)

    if st.session_state.results is None:
        st.error("No results available. Please run the analysis first.")
        if st.button("Back to Settings"):
            st.session_state.step = 5
            st.rerun()
        return

    results = st.session_state.results
    utility_round_decimals = 4
    if results.get("is_loaded"):
        st.info("Precomputed results were loaded from file. Analysis was not rerun.")

    st.success("Analysis complete.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Utilities", "Marginal Utilities", "Rankings", "Export", "Alternatives"]
    )

    with tab1:
        if results.get("utilities_df") is None:
            st.info("Utilities table was not included in the imported file.")
        else:
            utilities_df = results["utilities_df"].copy()
            utilities_df["Utility"] = pd.to_numeric(utilities_df["Utility"], errors="coerce").round(utility_round_decimals)
            utilities_df = utilities_df.sort_values("Utility", ascending=False)
            utilities_df["Rank"] = utilities_df["Utility"].rank(ascending=False, method="min").astype(int)
            st.caption(f"Predicted ranks use utility rounded to {utility_round_decimals} decimals.")
            st.dataframe(utilities_df[["Alternative", "Utility", "Rank", "Actual Rank"]], use_container_width=True, hide_index=True)

            fig = px.bar(
                utilities_df,
                x="Alternative",
                y="Utility",
                title="Alternative Utilities",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if results.get("criteria") is None:
            st.info("Detailed marginal utility charts are unavailable for imported precomputed results.")
            can_recompute = (
                st.session_state.alternatives_df is not None
                and bool(st.session_state.criteria_defs)
                and st.session_state.rankings is not None
            )
            if st.button(
                "Recompute Analysis",
                key="recompute_analysis_from_loaded_results",
                disabled=not can_recompute,
            ):
                with st.spinner("Recomputing analysis from current project data..."):
                    try:
                        st.session_state.results = run_uta_analysis(
                            alternatives_df=st.session_state.alternatives_df,
                            rankings=st.session_state.rankings,
                            criteria_defs=st.session_state.criteria_defs,
                            settings=st.session_state.algorithm_settings,
                            reference_names=st.session_state.reference_names,
                        )
                        st.session_state.model = st.session_state.results["model"]
                    except Exception as e:
                        st.error(f"Error recomputing analysis: {e}")
                        st.exception(e)
                    else:
                        st.success("Recomputed analysis with full marginal utility details.")
                        st.rerun()
            if not can_recompute:
                st.caption("To recompute, load or define criteria, alternatives, and rankings first.")
        else:
            partial_values = results.get("partial_values") or {}
            breakpoint_utilities = results.get("breakpoint_utilities") or {}
            for criterion in results["criteria"]:
                with st.expander(f"{criterion.name} ({criterion.type})", expanded=True):
                    criterion_values = partial_values.get(criterion.name) if isinstance(partial_values, dict) else None
                    utility_values = _get_breakpoint_utilities_for_view(
                        criterion=criterion,
                        criterion_values=criterion_values,
                        breakpoint_utilities=breakpoint_utilities,
                    )

                    if utility_values is None:
                        st.info(
                            "Exact breakpoint utilities are unavailable for this result payload. "
                            "Recompute the analysis to render the utility function accurately."
                        )
                        continue

                    mu_data = []
                    increments_to_next = [
                        float(utility_values[idx + 1] - utility_values[idx])
                        for idx in range(len(utility_values) - 1)
                    ] + [None]

                    for idx, bp in enumerate(criterion.breakpoints):
                        row = {
                            "Breakpoint": str(bp.position),
                            "Variable": bp.get_marginal_utility_var_name(),
                            "Utility at Breakpoint": float(utility_values[idx]),
                        }
                        if criterion.type != "nominal":
                            row["Increment to Next Breakpoint"] = increments_to_next[idx]
                        mu_data.append(row)

                    mu_df = pd.DataFrame(mu_data)

                    if criterion.type == "cardinal":
                        positions = [float(bp.position) for bp in criterion.breakpoints]
                        fig = px.line(
                            x=positions,
                            y=utility_values,
                            markers=True,
                            title=f"Utility Function: {criterion.name}",
                            labels={"x": criterion.name, "y": "Utility"},
                        )
                    else:
                        positions = [str(bp.position) for bp in criterion.breakpoints]
                        fig = px.bar(
                            x=positions,
                            y=utility_values,
                            title=f"Utility Function: {criterion.name}",
                            labels={"x": criterion.name, "y": "Utility"},
                        )

                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(mu_df, use_container_width=True, hide_index=True)

    with tab3:
        if results.get("utilities_df") is None:
            st.info("Ranking comparison is unavailable because utilities were not included in the imported file.")
        else:
            comparison_df = results["utilities_df"].copy()
            comparison_df["Utility"] = pd.to_numeric(comparison_df["Utility"], errors="coerce").round(utility_round_decimals)

            if "Reference" not in comparison_df.columns:
                comparison_df["Reference"] = True

            comparison_df["Predicted Rank (All)"] = (
                comparison_df["Utility"].rank(ascending=False, method="min").astype(int)
            )
            st.caption(f"Predicted ranks are computed from utility rounded to {utility_round_decimals} decimals.")

            reference_df = comparison_df[comparison_df["Reference"] == True].copy()
            non_reference_df = comparison_df[comparison_df["Reference"] == False].copy()

            st.markdown("### Reference alternatives (training set)")
            if reference_df.empty:
                st.info("No reference alternatives available.")
            else:
                reference_df["Predicted Rank (Reference)"] = (
                    reference_df["Utility"].rank(ascending=False, method="min").astype(int)
                )
                if "Actual Rank" in reference_df.columns:
                    actual_ref = pd.to_numeric(reference_df["Actual Rank"], errors="coerce")
                    reference_df["Actual Rank"] = actual_ref
                ref_cols = [
                    "Alternative",
                    "Actual Rank",
                    "Predicted Rank (Reference)",
                    "Predicted Rank (All)",
                    "Utility",
                ]
                st.dataframe(
                    reference_df[ref_cols].sort_values("Predicted Rank (Reference)", kind="stable"),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("### Non-reference alternatives")
            if non_reference_df.empty:
                st.info("No non-reference alternatives in this project.")
            else:
                non_reference_df["Predicted Rank (Non-reference)"] = (
                    non_reference_df["Utility"].rank(ascending=False, method="min").astype(int)
                )
                non_ref_cols = [
                    "Alternative",
                    "Predicted Rank (Non-reference)",
                    "Predicted Rank (All)",
                    "Utility",
                ]
                st.dataframe(
                    non_reference_df[non_ref_cols].sort_values("Predicted Rank (Non-reference)", kind="stable"),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("### All alternatives context")
            all_cols = ["Alternative", "Reference", "Actual Rank", "Predicted Rank (All)", "Utility"]
            st.dataframe(
                comparison_df[all_cols].sort_values("Predicted Rank (All)", kind="stable"),
                use_container_width=True,
                hide_index=True,
            )

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            if results.get("utilities_df") is not None:
                utilities_export_df = results["utilities_df"].copy()
                if "Rank" in utilities_export_df.columns:
                    utilities_export_df = utilities_export_df.drop(columns=["Rank"])
                st.download_button(
                    label="Download Utilities (CSV)",
                    data=utilities_export_df.to_csv(index=False),
                    file_name=f"{st.session_state.project_name}_utilities.csv",
                    mime="text/csv",
                )
            else:
                st.info("Utilities CSV is unavailable for this imported result payload.")
        with col2:
            alternatives_export = st.session_state.alternatives_df.copy()
            st.download_button(
                label="Download Alternatives (CSV)",
                data=alternatives_export.to_csv(index=False),
                file_name=f"{st.session_state.project_name}_alternatives.csv",
                mime="text/csv",
            )

        export_json = export_project_json(
            project_name=st.session_state.project_name,
            project_description=st.session_state.project_description,
            criteria_defs=st.session_state.criteria_defs,
            alternatives_df=st.session_state.alternatives_df,
            rankings=st.session_state.rankings,
            results=results,
            algorithm_settings=st.session_state.algorithm_settings,
            reference_names=st.session_state.reference_names,
        )
        st.download_button(
            label="Download Full Project (JSON)",
            data=export_json,
            file_name=f"{st.session_state.project_name.replace(' ', '_')}_full.json",
            mime="application/json",
        )

    with tab5:
        alternatives_df = st.session_state.alternatives_df
        if alternatives_df is None or alternatives_df.empty:
            st.info("No alternatives available in the current session.")
        else:
            alternatives_view = alternatives_df.copy()
            alternatives_view["Name"] = alternatives_view["Name"].astype(str)

            reference_names = [str(name) for name in (st.session_state.reference_names or [])]
            if (
                not reference_names
                and results.get("utilities_df") is not None
                and "Reference" in results["utilities_df"].columns
            ):
                reference_names = (
                    results["utilities_df"]
                    .loc[results["utilities_df"]["Reference"] == True, "Alternative"]
                    .astype(str)
                    .tolist()
                )

            if reference_names:
                reference_set = set(reference_names)
                alternatives_view.insert(1, "Reference", alternatives_view["Name"].isin(reference_set))
                if st.session_state.rankings is not None and len(st.session_state.rankings) == len(reference_names):
                    reference_rank_map = {
                        str(name): int(rank) for name, rank in zip(reference_names, st.session_state.rankings)
                    }
                    alternatives_view.insert(2, "Reference Rank", alternatives_view["Name"].map(reference_rank_map))

            if results.get("utilities_df") is not None:
                utility_cols = ["Alternative", "Utility"]
                utility_snapshot = results["utilities_df"].copy()
                utility_snapshot["Alternative"] = utility_snapshot["Alternative"].astype(str)
                utility_snapshot["Utility"] = pd.to_numeric(
                    utility_snapshot["Utility"], errors="coerce"
                ).round(utility_round_decimals)
                utility_snapshot["Predicted Rank"] = (
                    utility_snapshot["Utility"].rank(ascending=False, method="min").astype(int)
                )
                utility_cols.append("Predicted Rank")
                if "Actual Rank" in utility_snapshot.columns:
                    utility_cols.append("Actual Rank")
                alternatives_view = alternatives_view.merge(
                    utility_snapshot[utility_cols].rename(columns={"Alternative": "Name"}),
                    on="Name",
                    how="left",
                )

            sort_columns = ["Name"]
            ascending = [True]
            if "Reference Rank" in alternatives_view.columns:
                sort_columns = ["Reference", "Reference Rank", "Name"]
                ascending = [False, True, True]
            elif "Reference" in alternatives_view.columns:
                sort_columns = ["Reference", "Name"]
                ascending = [False, True]
            alternatives_view = alternatives_view.sort_values(
                by=sort_columns,
                ascending=ascending,
                kind="stable",
                na_position="last",
            )

            st.markdown("### Alternatives Properties")
            st.dataframe(alternatives_view, use_container_width=True, hide_index=True)

    st.markdown("---")
    c1, c2, _ = st.columns(3)
    with c1:
        if st.button("Back to Summary", use_container_width=True):
            st.session_state.step = 6
            st.rerun()
    with c2:
        if st.button("New Project", use_container_width=True):
            reset_state()
            st.rerun()


def render_summary():
    st.markdown(f"**Project:** {st.session_state.project_name}")
    if st.session_state.project_description:
        st.caption(st.session_state.project_description)
    st.markdown("### Summary")

    settings = st.session_state.algorithm_settings
    alts_df = st.session_state.alternatives_df
    ref_names = st.session_state.reference_names or (alts_df["Name"].tolist() if alts_df is not None else [])
    ref_names = [n for n in ref_names if n in (alts_df["Name"].tolist() if alts_df is not None else [])]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Criteria**")
        for name, spec in st.session_state.criteria_defs.items():
            st.write(f"- {name} ({spec.get('type', '?')})")
        st.markdown("**Algorithm Settings**")
        st.write(f"Algorithm: {settings.get('algorithm', 'UTASTAR')}")
        st.write(f"Sigma: {float(settings.get('sigma', 0.001)):.4f}")
        st.write(f"Cardinal breakpoints: {settings.get('breakpoints', 'quantile')}")
        st.write(f"Missing value treatment: {settings.get('missing_value_treatment', 'assumeAverageValue')}")

    with col2:
        st.markdown("**Alternatives & Rankings**")
        n_total = len(alts_df) if alts_df is not None else 0
        st.write(f"Total alternatives: {n_total}")
        st.write(f"Reference alternatives: {len(ref_names)}")

        rankings = st.session_state.rankings
        if rankings is not None and alts_df is not None and len(rankings) == len(ref_names):
            rank_df = pd.DataFrame({"Alternative": ref_names, "Rank": rankings}).sort_values("Rank", kind="stable")
            st.dataframe(rank_df, use_container_width=True, hide_index=True)
        elif rankings is None:
            st.warning("No rankings defined yet. Please complete the Rank Preferences step.")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Back", use_container_width=True):
            st.session_state.step = 5
            st.rerun()
    with c2:
        can_run = st.session_state.rankings is not None
        if st.button("Run Analysis", use_container_width=True, type="primary", disabled=not can_run):
            with st.spinner("Running UTA optimization..."):
                try:
                    st.session_state.results = run_uta_analysis(
                        alternatives_df=st.session_state.alternatives_df,
                        rankings=st.session_state.rankings,
                        criteria_defs=st.session_state.criteria_defs,
                        settings=st.session_state.algorithm_settings,
                        reference_names=st.session_state.reference_names,
                    )
                    st.session_state.model = st.session_state.results["model"]
                except Exception as e:
                    st.error(f"Error running analysis: {e}")
                    st.exception(e)
                    return
            st.session_state.step = 7
            st.rerun()
    with c3:
        if st.session_state.results is not None:
            if st.button("Jump to Results", use_container_width=True):
                st.session_state.step = 7
                st.rerun()


def _load_example(example_name: str) -> None:
    X, y, criteria_defs = load_example_dataset(example_name)
    st.session_state.project_name = f"Example: {example_name.title()}"
    st.session_state.project_description = f"Built-in {example_name} example dataset."
    st.session_state.criteria_defs = criteria_defs
    X = X.reset_index()
    X = X.rename(columns={X.columns[0]: "Name"})
    validate_alternatives_schema(X, criteria_defs)
    st.session_state.alternatives_df = X
    st.session_state.rankings = list(y)
    st.session_state.reference_names = X["Name"].astype(str).tolist()
    st.session_state.ranking_groups = None
    st.session_state.step = 6
    st.success(f"Loaded {example_name} example")
    st.rerun()
