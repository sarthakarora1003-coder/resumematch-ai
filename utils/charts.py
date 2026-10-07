"""
charts.py
Plotly chart generators for the Streamlit dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def gauge_chart(score: int, title: str, color_thresholds: tuple = (40, 70)) -> go.Figure:
    """
    Create a gauge/speedometer chart for a score (0–100).
    color_thresholds: (low_cutoff, high_cutoff) — below low = red, above high = green
    """
    low, high = color_thresholds

    if score < low:
        color = "#ef4444"   # red
    elif score < high:
        color = "#f59e0b"   # amber
    else:
        color = "#22c55e"   # green

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title, "font": {"size": 16}},
        number={"suffix": "%", "font": {"size": 32}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "white",
            "steps": [
                {"range": [0, low], "color": "#fee2e2"},
                {"range": [low, high], "color": "#fef3c7"},
                {"range": [high, 100], "color": "#dcfce7"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def skills_bar_chart(matched: list[str], missing_required: list[str], missing_preferred: list[str]) -> go.Figure:
    """Horizontal bar chart showing skill match breakdown."""
    categories = ["Matched Skills", "Missing Required", "Missing Preferred"]
    counts = [len(matched), len(missing_required), len(missing_preferred)]
    colors = ["#22c55e", "#ef4444", "#f59e0b"]

    fig = go.Figure(go.Bar(
        y=categories,
        x=counts,
        orientation="h",
        marker_color=colors,
        text=counts,
        textposition="outside",
    ))
    fig.update_layout(
        title="Skill Match Breakdown",
        xaxis_title="Number of Skills",
        height=220,
        margin=dict(l=10, r=40, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def radar_chart(scores: dict[str, int]) -> go.Figure:
    """Radar/spider chart for multi-dimension scoring."""
    categories = list(scores.keys())
    values = list(scores.values())
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(99, 102, 241, 0.2)",
        line=dict(color="#6366f1", width=2),
        marker=dict(size=6, color="#6366f1"),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
        ),
        showlegend=False,
        title="Match Radar",
        height=300,
        margin=dict(l=40, r=40, t=50, b=10),
    )
    return fig


def skills_bubble_chart(matched: list[str], missing: list[str]) -> go.Figure:
    """Scatter/bubble chart showing matched vs missing skills."""
    data = []
    for skill in matched:
        data.append({"Skill": skill, "Status": "✅ Matched", "x": 1})
    for skill in missing:
        data.append({"Skill": skill, "Status": "❌ Missing", "x": 0})

    if not data:
        return go.Figure()

    df = pd.DataFrame(data)
    fig = px.strip(
        df,
        x="x",
        y="Skill",
        color="Status",
        color_discrete_map={"✅ Matched": "#22c55e", "❌ Missing": "#ef4444"},
        title="Skills Overview",
    )
    fig.update_layout(
        height=max(300, len(data) * 25),
        xaxis=dict(tickvals=[0, 1], ticktext=["Missing", "Matched"], title=""),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
