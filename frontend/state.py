"""Frontend session-state helpers and workflow metadata."""

import streamlit as st

STEP_TITLES = [
    "Home",
    "Project Setup",
    "Algorithm Settings",
    "Define Criteria",
    "Add Alternatives",
    "Rank Preferences",
    "Summary",
    "Results",
]


def init_state() -> None:
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "project_name" not in st.session_state:
        st.session_state.project_name = None
    if "project_description" not in st.session_state:
        st.session_state.project_description = ""
    if "criteria_defs" not in st.session_state:
        st.session_state.criteria_defs = {}
    if "alternatives_df" not in st.session_state:
        st.session_state.alternatives_df = None
    if "rankings" not in st.session_state:
        st.session_state.rankings = None
    if "reference_names" not in st.session_state:
        st.session_state.reference_names = None
    if "ranking_groups" not in st.session_state:
        st.session_state.ranking_groups = None
    if "model" not in st.session_state:
        st.session_state.model = None
    if "results" not in st.session_state:
        st.session_state.results = None
    if "algorithm_settings" not in st.session_state:
        st.session_state.algorithm_settings = {
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


def reset_state() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
