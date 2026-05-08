"""
ui.pages.data_upload – Data Upload & Analysis page.
"""

from __future__ import annotations

import streamlit as st

from src import config
from src.data_processor import auto_process_data, get_dataset_summary, load_data, validate_dataset
from src.visualisations import correlation_heatmap, feature_distributions


def render() -> None:
    """Render the Data Upload page."""
    st.title("📤 Data Upload & Analysis")

    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file",
        type=config.SUPPORTED_FILE_TYPES,
        help="Supported formats: .csv, .xlsx, .xls",
    )

    if uploaded_file is not None:
        with st.spinner("Loading dataset…"):
            raw = load_data(uploaded_file.read(), uploaded_file.name)

        is_valid, msg = validate_dataset(raw)
        if not is_valid:
            st.error(f"❌ {msg}")
        else:
            with st.spinner("Auto-processing data…"):
                processed, encoders = auto_process_data(raw)

            st.session_state.raw_data       = raw
            st.session_state.processed_data = processed
            st.session_state.label_encoders = encoders
            st.session_state.model          = None
            st.session_state.model_name     = None
            st.session_state.features       = []
            st.session_state.target         = None
            st.session_state.X_test         = None
            st.session_state.y_test         = None
            st.session_state.auto_results_df = None
            st.session_state.fitted_models  = {}

            st.success("✅ Dataset loaded and processed successfully!")

            summary = get_dataset_summary(raw)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Rows",          f"{summary['rows']:,}")
            c2.metric("Columns",       summary["columns"])
            c3.metric("Missing Values",summary["missing_values"])
            c4.metric("Duplicate Rows",summary["duplicate_rows"])
            c5.metric("Numeric Cols",  len(summary["numeric_cols"]))

            st.markdown("---")

            tab_raw, tab_proc, tab_stats, tab_corr, tab_dist = st.tabs([
                "Raw Data", "Processed Data", "Statistics",
                "Correlation Heatmap", "Feature Distributions",
            ])

            with tab_raw:
                st.dataframe(raw, use_container_width=True, height=380)

            with tab_proc:
                st.caption("All categoricals label-encoded; missing values imputed.")
                st.dataframe(processed, use_container_width=True, height=380)

            with tab_stats:
                st.dataframe(processed.describe().round(3).T, use_container_width=True)

            with tab_corr:
                if len(summary["numeric_cols"]) >= 2:
                    st.pyplot(correlation_heatmap(processed[summary["numeric_cols"]]))
                else:
                    st.info("Need at least 2 numeric columns for correlation heatmap.")

            with tab_dist:
                st.plotly_chart(
                    feature_distributions(processed),
                    use_container_width=True,
                )
