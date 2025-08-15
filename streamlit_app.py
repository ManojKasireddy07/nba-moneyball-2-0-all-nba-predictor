"""
Streamlit dashboard for NBA Moneyball 2.0.

This application exposes three pages:

1. **Model Insights** – visualizes global feature importance and displays
   evaluation metrics.
2. **Player Profile Explorer** – allows users to select a player and season
   and compare their key statistics against the All‑NBA averages for that
   season using a radar chart.
3. **Future Stars Predictor** – lists players from the most recent season
   who did not make an All‑NBA team along with their predicted probability
   of doing so next year and plots these probabilities against VORP.

To run locally use:

    streamlit run src/app/streamlit_app.py
"""
from __future__ import annotations

import json
import pandas as pd
import plotly.express as px
import streamlit as st

from .utils import (
    load_master,
    load_metrics,
    load_global_importance,
    load_predictions,
    get_player_row,
    compute_all_nba_average,
)


def page_model_insights() -> None:
    st.header("Model Insights")
    shap_df = load_global_importance()
    metrics = load_metrics()
    # Bar chart of top 15 features
    if not shap_df.empty:
        fig = px.bar(
            shap_df.head(15),
            x="mean_abs_contrib",
            y="feature",
            orientation="h",
            title="Global Feature Importance (Top 15)",
            labels={"mean_abs_contrib": "Mean |Contribution|", "feature": "Feature"},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Global feature importance not available. Run shap_analysis.py to generate it.")
    # Display aggregated metrics
    if metrics:
        st.subheader("Aggregated Performance Metrics (2010–Latest)")
        metrics_df = pd.DataFrame(metrics).T
        st.table(metrics_df)
    else:
        st.info("Metrics summary not found. Run evaluate.py to generate it.")


def page_player_profile() -> None:
    st.header("Player Profile Explorer")
    df = load_master()
    seasons = sorted(df["season_end_year"].unique())
    selected_season = st.selectbox("Season (end year)", seasons, index=len(seasons) - 1)
    season_df = df[df["season_end_year"] == selected_season]
    players = sorted(season_df["name"].unique())
    selected_player = st.selectbox("Player", players)
    player_row = get_player_row(df, selected_player, selected_season)
    if player_row is None:
        st.warning("Player data not available.")
        return
    # Define stats to display
    stat_cols = []
    # Use common stats if present
    for col in ["PTS", "TRB", "AST", "VORP", "PER"]:
        if col in df.columns:
            stat_cols.append(col)
    st.write(f"### {selected_player} – {selected_season} season")
    # Show raw stats table
    st.write(player_row[stat_cols].to_frame().T)
    # Radar chart comparing to All‑NBA average
    if stat_cols:
        nba_avg = compute_all_nba_average(df, selected_season, stat_cols)
        radar_df = pd.DataFrame(
            {
                "stat": stat_cols,
                selected_player: [player_row[col] for col in stat_cols],
                "All‑NBA Avg": [nba_avg[col] for col in stat_cols],
            }
        )
        fig = px.line_polar(
            radar_df,
            r=[player_row[col] for col in stat_cols] + [nba_avg[col] for col in stat_cols],
            theta=stat_cols + stat_cols,
            color=[selected_player] * len(stat_cols) + ["All‑NBA Avg"] * len(stat_cols),
            line_close=True,
            title="Player vs All‑NBA Average",
        )
        st.plotly_chart(fig, use_container_width=True)


def page_future_stars() -> None:
    st.header("Future Stars Predictor")
    df = load_master()
    preds = load_predictions()
    if preds.empty:
        st.info("Prediction probabilities not found. Run train.py to generate them.")
        return
    # Determine the most recent season
    recent_year = preds["season_end_year"].max()
    st.write(f"Most recent season: {recent_year}")
    recent_df = preds[preds["season_end_year"] == recent_year].copy()
    # Merge with master to get All‑NBA status and VORP
    merged = recent_df.merge(
        df[["name", "season_end_year", "was_all_nba", "VORP"]],
        on=["name", "season_end_year"],
        how="left",
    )
    # Filter players who did not make All‑NBA
    candidates = merged[merged["was_all_nba"] == 0]
    # Top 20 by probability
    top20 = candidates.nlargest(20, "probability")[
        ["name", "team", "probability", "VORP"]
    ].reset_index(drop=True)
    st.subheader("Top 20 Breakout Candidates")
    st.write(top20.style.format({"probability": "{:.2%}", "VORP": "{:.2f}"}))
    # Scatter plot: probability vs VORP colored by actual All‑NBA status
    fig = px.scatter(
        merged,
        x="probability",
        y="VORP",
        color=merged["was_all_nba"].map({0: "Not All‑NBA", 1: "All‑NBA"}),
        hover_name="name",
        title="Predicted Probability vs VORP",
        labels={"probability": "Predicted Probability", "VORP": "Value Over Replacement"},
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="NBA Moneyball 2.0", layout="wide")
    page = st.sidebar.radio(
        "Navigate",
        ["Model Insights", "Player Profile Explorer", "Future Stars Predictor"],
    )
    if page == "Model Insights":
        page_model_insights()
    elif page == "Player Profile Explorer":
        page_player_profile()
    else:
        page_future_stars()


if __name__ == "__main__":
    main()