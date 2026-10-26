"""Plotly figures render actual observations with explicit units and periods."""
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

TEAL = "#B8F275"
CORAL = "#ED9B96"
BLUE = "#91B5CE"


def style_figure(figure: go.Figure, title: str, y_title: str = "") -> go.Figure:
    figure.update_layout(template="plotly_dark", title=dict(text=title, font=dict(size=13)), height=320,
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         margin=dict(l=8, r=8, t=45 if title else 20, b=15),
                         font=dict(family="Arial, sans-serif", color="#96A498", size=11),
                         legend=dict(orientation="h", y=1.06, font=dict(size=10)),
                         yaxis_title=y_title, hovermode="x unified",
                         hoverlabel=dict(bgcolor="#263329", font_color="#EDF2E9"))
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="rgba(150,170,150,.09)", zeroline=False)
    return figure


def price_chart(frame: pd.DataFrame, indicators=(), anomalies=None, kind="Line") -> go.Figure:
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03, row_heights=[0.82, 0.18])
    if kind == "Candlestick":
        figure.add_trace(go.Candlestick(x=frame.Date, open=frame.Open, high=frame.High,
                                        low=frame.Low, close=frame.Close, name="OHLC",
                                        increasing_line_color=TEAL, decreasing_line_color=CORAL), row=1, col=1)
    else:
        figure.add_trace(go.Scatter(x=frame.Date, y=frame.Close, name="Close",
                                    line=dict(color=TEAL, width=2)), row=1, col=1)
    for name, color in zip(indicators, [BLUE, "#D8AEFF", "#F4C66A"]):
        figure.add_trace(go.Scatter(x=frame.Date, y=frame[name], name=name.upper(),
                                    line=dict(color=color, width=1)), row=1, col=1)
    colors = ["rgba(184,242,117,.25)" if c >= o else "rgba(237,155,150,.25)"
              for c, o in zip(frame.Close, frame.Open)]
    figure.add_trace(go.Bar(x=frame.Date, y=frame.Volume, name="Volume", marker_color=colors,
                            showlegend=False), row=2, col=1)
    if anomalies is not None:
        outliers = anomalies.loc[anomalies["anomaly"] & anomalies.Date.between(frame.Date.min(), frame.Date.max())]
        figure.add_trace(go.Scatter(x=outliers.Date, y=outliers.Close, mode="markers",
                                    name="Outlier", marker=dict(color=CORAL, size=8)), row=1, col=1)
    style_figure(figure, "")
    figure.update_layout(height=420, xaxis_rangeslider_visible=False, yaxis2_title="Volume")
    figure.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    figure.update_yaxes(side="right", tickformat=",.0f", row=1, col=1)
    figure.update_yaxes(side="right", tickformat=".2s", showgrid=False, row=2, col=1)
    return figure


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
