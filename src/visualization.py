"""Plotly figures render actual observations with explicit units and periods."""
import plotly.graph_objects as go
import pandas as pd

TEAL = "#44D7B6"
CORAL = "#FF7C8B"
BLUE = "#7DABFF"


def style_figure(figure: go.Figure, title: str, y_title: str = "") -> go.Figure:
    figure.update_layout(template="plotly_dark", title=title, height=420,
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         margin=dict(l=15, r=15, t=55, b=25),
                         legend=dict(orientation="h", y=1.10),
                         yaxis_title=y_title, hovermode="x unified")
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="rgba(150,170,200,.12)")
    return figure


def price_chart(frame: pd.DataFrame, indicators=(), anomalies=None) -> go.Figure:
    figure = go.Figure(go.Scatter(x=frame.Date, y=frame.Close, name="Historical close",
                                 line=dict(color=TEAL, width=2)))
    for name, color in zip(indicators, [BLUE, "#D8AEFF", "#F4C66A"]):
        figure.add_trace(go.Scatter(x=frame.Date, y=frame[name], name=name.upper(),
                                    line=dict(color=color, width=1)))
    if anomalies is not None:
        outliers = anomalies.loc[anomalies["anomaly"]]
        figure.add_trace(go.Scatter(x=outliers.Date, y=outliers.Close, mode="markers",
                                    name="Statistical outlier", marker=dict(color=CORAL, size=8)))
    return style_figure(figure, "Historical price", "Price · source units")


def series_chart(frame: pd.DataFrame, column: str, title: str, percent=False) -> go.Figure:
    figure = go.Figure(go.Scatter(x=frame.Date, y=frame[column], name=title,
                                 line=dict(color=BLUE, width=2), fill="tozeroy"))
    style_figure(figure, title)
    if percent:
        figure.update_yaxes(tickformat=".1%")
    return figure


def confusion_chart(matrix: list, title: str) -> go.Figure:
    names = ["Bearish", "Bullish"]
    figure = go.Figure(go.Heatmap(z=matrix, x=names, y=names, colorscale="Teal",
                                 text=matrix, texttemplate="%{text}", showscale=False))
    style_figure(figure, title, "Actual")
    figure.update_xaxes(title="Predicted")
    return figure
