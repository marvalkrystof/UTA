"""UTA Frontend - Streamlit Application entrypoint."""

import os
import sys

# Ensure project root is importable when running from frontend/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from state import STEP_TITLES, init_state
from views import (
    render_add_alternatives,
    render_algorithm_settings,
    render_define_criteria,
    render_home,
    render_project_setup,
    render_rank_preferences,
    render_results,
    render_summary,
)


st.set_page_config(
    page_title="UTA Preference Learning",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _step_requirement_error(target_step: int) -> str | None:
    project_name = st.session_state.project_name
    criteria_defs = st.session_state.criteria_defs or {}
    alternatives_df = st.session_state.alternatives_df
    rankings = st.session_state.rankings
    results = st.session_state.results

    if target_step >= 1 and not project_name:
        return "Please complete Project Setup first."

    if target_step >= 4:
        if not criteria_defs:
            return "Please define at least one criterion first."
        unresolved = [name for name, spec in criteria_defs.items() if spec.get("type") is None]
        if unresolved:
            return f"Please resolve criterion types first: {', '.join(unresolved)}"

    if target_step >= 5:
        if alternatives_df is None or len(alternatives_df) < 2:
            return "Please add at least 2 alternatives first."

    if target_step >= 6:
        if rankings is None or len(rankings) < 2:
            return "Please finish ranking reference alternatives first."

    if target_step >= 7 and results is None:
        return "Please run analysis first."

    return None


def _render_step_dots(current_step: int, max_step: int) -> None:
    if current_step == 0:
        return

    warning_message = None
    cols = st.columns(len(STEP_TITLES) - 1)
    for step_num in range(1, len(STEP_TITLES)):
        with cols[step_num - 1]:
            is_current = step_num == current_step
            label = "⬤" if is_current else "◯"
            button_help = f"Step {step_num}: {STEP_TITLES[step_num]}"
            if st.button(
                label,
                key=f"dot_nav_{step_num}",
                use_container_width=True,
                help=button_help,
                type="tertiary",
            ):
                if step_num <= current_step:
                    st.session_state.step = step_num
                    st.rerun()

                error = _step_requirement_error(step_num)
                if error:
                    warning_message = error
                else:
                    st.session_state.step = step_num
                    st.rerun()

    label_cols = st.columns(len(STEP_TITLES) - 1)
    with label_cols[current_step - 1]:
        st.markdown(
            (
                "<div style='text-align:center; margin-top:-0.25rem; font-size:0.82rem; opacity:0.82;'>"
                f"Workflow progress: Step {current_step}/{max_step} - {STEP_TITLES[current_step]}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    if warning_message:
        st.warning(warning_message)

def _render_progress() -> None:
    current_step = int(st.session_state.step)
    if current_step == 0:
        return
    max_step = len(STEP_TITLES) - 1
    if max_step <= 1:
        fill_pct = 100.0
    else:
        fill_pct = 100.0 * max(0, current_step - 1) / max(1, max_step - 1)

    st.markdown(
        f"""
        <div style='margin: 0.15rem 0 0.2rem 0;'>
            <div style='height: 8px; width: 100%; background: rgba(151, 166, 195, 0.30); border-radius: 999px; overflow: hidden;'>
                <div style='height: 100%; width: {fill_pct:.2f}%; background: rgba(255, 75, 75, 0.95); border-radius: 999px; transition: width 0.2s ease;'></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_step_dots(current_step=current_step, max_step=max_step)


def main() -> None:
    init_state()
    _render_progress()

    step = st.session_state.step
    if step == 0:
        render_home()
    elif step == 1:
        render_project_setup()
    elif step == 2:
        render_algorithm_settings()
    elif step == 3:
        render_define_criteria()
    elif step == 4:
        render_add_alternatives()
    elif step == 5:
        render_rank_preferences()
    elif step == 6:
        render_summary()
    elif step == 7:
        render_results()
    else:
        st.error("Unknown workflow step. Returning to home.")
        st.session_state.step = 0
        st.rerun()


if __name__ == "__main__":
    main()

