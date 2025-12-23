# figures.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data_access import radar_values

def radar_figure(p1_id, p2_id, start_date, end_date, level):
    # if not p2_id or p2_id == "(All)":
    #     fig = go.Figure()
    #     fig.update_layout(annotations=[dict(text="Select Player 2 to compare", x=0.5, y=0.5, showarrow=False)])
    #     return fig

    n1, labels, v1 = radar_values(p1_id, start_date, end_date, level)
    
    if labels is None:
        fig = go.Figure()
        fig.update_layout(annotations=[dict(text="No data for selected players", x=0.5, y=0.5, showarrow=False)])
        return fig

    # close loop
    labels2 = labels + [labels[0]]
    v1_ = v1 + [v1[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=v1_, theta=labels2, fill="toself", name=f"{n1} (P1)"))

    if p2_id != "(All)" and p2_id is not None:
        n2, _, v2 = radar_values(p2_id, start_date, end_date, level)
        v2_ = v2 + [v2[0]]
        fig.add_trace(go.Scatterpolar(r=v2_, theta=labels2, fill="toself", name=f"{n2} (P2)"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=360,
        showlegend=True
    )
    return fig


def stacked_surface_fig(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            annotations=[dict(text="No data after filters", x=0.5, y=0.5, showarrow=False)]
        )
        return fig

    fig = go.Figure()
    fig.add_bar(
        x=df["tournament_surface"], y=df["wins"], name="Wins", marker_color="#1ade1a",
        customdata=np.stack([df["win_rate"].values, df["wins"].values, df["losses"].values], axis=1),
        hovertemplate="Surface=%{x}<br>Wins=%{y}<br>Win rate=%{customdata[0]:.0%}<extra></extra>"
    )
    fig.add_bar(
        x=df["tournament_surface"], y=df["losses"], name="Losses", marker_color="#f70707",
        customdata=1-np.stack([df["win_rate"].values, df["wins"].values, df["losses"].values], axis=1),
        hovertemplate="Surface=%{x}<br>Losses=%{y}<br>Loss rate=%{customdata[0]:.0%}<extra></extra>"
    )

    fig.update_layout(
        barmode="stack",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h"),
        hovermode="x unified"
    )
    return fig

def stacked_wl_fig(df, xcol, title=""):
    if df.empty:
        fig = go.Figure()
        fig.update_layout(annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False)])
        return fig

    wins = df[df["result"] == "Win"]
    losses = df[df["result"] == "Loss"]

    fig = go.Figure()
    fig.add_bar(x=wins[xcol], y=wins["cnt"], name="Win")
    fig.add_bar(x=losses[xcol], y=losses["cnt"], name="Loss")

    fig.update_layout(barmode="stack", margin=dict(l=10, r=10, t=30, b=10), title=title, hovermode="x unified")
    return fig

def double_donut_win_round(df: pd.DataFrame) -> go.Figure:
    """
    Inner ring: Win / Loss
    Outer ring: Round counts under Win/Loss
    df columns required: result (Win/Loss), round, cnt
    """
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="No round data", x=0.5, y=0.5, showarrow=False)],
            margin=dict(l=10, r=10, t=30, b=10),
        )
        return fig

    df = df.copy()
    df["result"] = df["result"].astype(str)
    df["round"] = df["round"].astype(str)
    df["cnt"] = pd.to_numeric(df["cnt"], errors="coerce").fillna(0).astype(int)

    # Ensure only Win/Loss categories
    df = df[df["result"].isin(["Win", "Loss"])]
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="No Win/Loss round data", x=0.5, y=0.5, showarrow=False)],
            margin=dict(l=10, r=10, t=30, b=10),
        )
        return fig

    labels = []
    parents = []
    values = []

    # --- Inner ring nodes ---
    totals = df.groupby("result", as_index=False)["cnt"].sum()

    for r in ["Win", "Loss"]:
        v = int(totals.loc[totals["result"] == r, "cnt"].sum())
        if v > 0:
            labels.append(r)
            parents.append("")   # root nodes -> inner ring
            values.append(v)

    # --- Outer ring nodes (round breakdown under each result) ---
    # IMPORTANT: label must be unique across all nodes
    # so we prefix with result
    for row in df.itertuples(index=False):
        child_label = f"{row.result} · {row.round}"
        labels.append(child_label)
        parents.append(row.result)     # parent is "Win" or "Loss"
        values.append(int(row.cnt))

    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        maxdepth=2,
        hovertemplate="%{label}<br>Count=%{value}<extra></extra>",
    ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig