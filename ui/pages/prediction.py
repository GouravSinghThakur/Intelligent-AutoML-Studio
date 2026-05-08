"""
ui.pages.prediction – Prediction page.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from ui.helpers import require_model


def render() -> None:
    """Render the Prediction page."""
    st.title("🔮 Make Predictions")

    if not require_model():
        st.stop()

    model      = st.session_state.model
    features   = st.session_state.features
    target     = st.session_state.target
    label_enc  = st.session_state.label_encoders
    model_name = st.session_state.model_name
    task_type  = st.session_state.task_type

    task_label = "Classification" if task_type == config.TASK_CLASSIFICATION else "Regression"
    st.info(f"🧠 Active model: **{model_name}** | Task: **{task_label}**")
    st.markdown("---")
    st.subheader("Enter feature values")

    input_data = {}
    cols = st.columns(min(3, len(features)))
    for i, feature in enumerate(features):
        col = cols[i % len(cols)]
        with col:
            if feature in label_enc:
                options = list(label_enc[feature].classes_)
                selected = st.selectbox(f"**{feature}**", options, key=f"pred_{feature}")
                input_data[feature] = int(label_enc[feature].transform([selected])[0])
            else:
                default_val = float(
                    st.session_state.processed_data[feature].median()
                    if st.session_state.processed_data is not None else 0.0
                )
                input_data[feature] = st.number_input(
                    f"**{feature}**",
                    value=default_val,
                    key=f"pred_{feature}",
                    format="%.4f",
                )

    st.markdown("---")

    predict_btn_col, _ = st.columns([1, 4])
    with predict_btn_col:
        predict_clicked = st.button("🔮 Predict", type="primary", use_container_width=True)

    if predict_clicked:
        input_df = pd.DataFrame([input_data])

        try:
            prediction = model.predict(input_df)
            raw_pred   = prediction[0]

            if task_type == config.TASK_CLASSIFICATION:
                if target in label_enc:
                    display_pred = label_enc[target].inverse_transform([int(raw_pred)])[0]
                else:
                    display_pred = raw_pred

                st.success(f"### Predicted **{target}**: `{display_pred}`")

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_df)[0]
                    classes = (
                        list(label_enc[target].classes_)
                        if target in label_enc
                        else [str(c) for c in model.classes_]
                    )
                    proba_df = pd.DataFrame({"Class": classes, "Probability": proba}).sort_values(
                        "Probability", ascending=False
                    )

                    st.subheader("📊 Prediction Probabilities")
                    st.dataframe(proba_df.style.format({"Probability": "{:.2%}"}), use_container_width=True)

                    fig = px.bar(
                        proba_df,
                        x="Class",
                        y="Probability",
                        color="Probability",
                        color_continuous_scale=["#6C63FF", "#FF6584"],
                        title="Prediction Probability Distribution",
                        template="plotly_dark",
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        coloraxis_showscale=False,
                        yaxis=dict(tickformat=".0%"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.success(f"### Predicted **{target}**: `{raw_pred:.4f}`")

        except Exception as exc:
            st.error(f"❌ Prediction failed: {exc}")
            st.exception(exc)
