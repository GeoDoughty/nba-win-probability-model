"""Plots to evaluate model performance"""

"""Basic script for some simple model evals. These include:
1. Accuracy
2. Predictive power (how confident is the model in itself)
3. Accuracy over game time?

Could add something around over accurate predictions.
Also what does a ROC curve show here?
"""
import polars as pl
import plotly.express as px
from eval.metrics import calculate_accuracy_metrics

PRED_COL = "home_win_prob"

pred_df = pl.read_parquet("./data/processed/logit/basic_train_22.parquet")

metrics = calculate_accuracy_metrics(pred_df, PRED_COL)

# According to this I can split by 0.005 probs
px.histogram(
    pred_df,
    x=PRED_COL,
    title="Distribution of Home Win Probabilities",
).write_html("home_win_prob_dist.html")

prob_rolling_df = (
    pred_df.with_columns(pl.col(PRED_COL).floordiv(0.005).alias("prob_bin"))
    .group_by("prob_bin")
    .agg(
        pl.col("home_win").cast(pl.Boolean).mean(),
        pl.col("home_win").len().alias("count"),
    )
    .with_columns(pl.col("prob_bin") * 0.005)
    .sort("prob_bin")
)

fig = px.line(
    prob_rolling_df,
    x="prob_bin",
    y="home_win",
    title="Average Home Win Probability by Probability Bin",
    hover_data=["count"],
)
fig.add_scatter(
    x=prob_rolling_df["prob_bin"],
    y=prob_rolling_df["prob_bin"],
    mode="lines",
)
fig.write_html("home_win_prob_rolling.html")


px.line(
    pred_df.filter(pl.col("GAME_ID") == "0022201052"),
    x="gametime_elapsed",
    y=PRED_COL,
).write_html("yes_lawd.html")
