# tabs/backtest_tab.py
from dash import dcc, html, Input, Output, State, callback_context, ALL
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc

from .backtest_utils import TwoSidedKellyBacktester, TopPlayerKellyBacktester, TwoSidedBacktester, trades_df, fig_profit_by, fig_roi_by_rankdiff, fig_pnl_by_prob_bucket

def read_output_model():
    test_df = pd.read_csv('test_df.csv')
    return test_df

def make_backtest_layout():
    # NOTE: All IDs must be unique in your whole app.
    # I prefix everything with "bt_" to avoid collision with your current tabs.
    return dbc.Container(fluid=True, style={"padding": "18px"}, children=[
        dcc.Store(id="bt_params_store"),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Backtest Strategy", style={"fontSize": "18px", "fontWeight": "600", "marginBottom": "10px"}),
                dcc.Dropdown(
                    id="bt_strategy",
                    options=[
                        {"label": "Two-Sided Kelly", "value": "kelly"},
                        {"label": "Top Player Kelly", "value": "top_player"},
                        {"label": "Two-Sided Simple", "value": "simple"},
                    ],
                    value="kelly",
                    clearable=False,
                ),
                html.Div(id="bt_params_panel", style={"marginTop": "12px"}),
                html.Button("🚀 Run Backtest", id="bt_run", n_clicks=0,
                            style={"width": "100%", "marginTop": "12px", "padding": "12px",
                                   "borderRadius": "8px", "border": "none",
                                   "backgroundColor": "#27ae60", "color": "white",
                                   "fontWeight": "700"}),
            ])), width=4),

            dbc.Col(dbc.Card(dbc.CardBody([
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.Div("Profit", style={"fontSize": "12px", "opacity": 0.8}),
                        html.Div(id="bt_kpi_profit", style={"fontSize": "26px", "fontWeight": 700}),
                    ])), width=3),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.Div("ROI", style={"fontSize": "12px", "opacity": 0.8}),
                        html.Div(id="bt_kpi_roi", style={"fontSize": "26px", "fontWeight": 700}),
                    ])), width=3),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.Div("Final capital", style={"fontSize": "12px", "opacity": 0.8}),
                        html.Div(id="bt_kpi_final", style={"fontSize": "26px", "fontWeight": 700}),
                    ])), width=3),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.Div("Max DD", style={"fontSize": "12px", "opacity": 0.8}),
                        html.Div(id="bt_kpi_dd", style={"fontSize": "26px", "fontWeight": 700}),
                    ])), width=3),
                ], className="g-2"),
                html.Div(style={"height": "12px"}),
                dcc.Loading(dcc.Graph(id="bt_equity", config={"displayModeBar": False}, style={"height": "420px"})),
                html.Div(id="bt_stats_table", style={"marginTop": "10px"}),
            ])), width=8),
        ], className="g-3"),

        html.Div(style={"height": "14px"}),
        
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Profit by Tournament Type", style={"fontWeight": 600, "marginBottom": 8}),
                dcc.Graph(id="bt_profit_by_type", config={"displayModeBar": False}, style={"height": "320px"})
            ])), width=6),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("ROI by Rank Difference", style={"fontWeight": 600, "marginBottom": 8}),
                dcc.Graph(id="bt_roi_by_rankdiff", config={"displayModeBar": False}, style={"height": "320px"})
            ])), width=6),
        ], className="g-3"),

        html.Div(style={"height": "14px"}),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("ROI by Probability Bucket", style={"fontWeight": 600, "marginBottom": 8}),
                dcc.Graph(id="bt_pnl_by_prob_bucket", config={"displayModeBar": False}, style={"height": "320px"})
            ])), width=12),
        ], className="g-3"),
    ])


def register_backtest_callbacks(app, load_backtest_df_func):
    """
    load_backtest_df_func: a function you provide that returns the dataframe used for backtest.
      Example signature:
        df = load_backtest_df_func()
      or
        df = load_backtest_df_func(strategy=..., other_filters=...)
    """

    @app.callback(
        Output("bt_params_panel", "children"),
        Input("bt_strategy", "value"),
    )
    def bt_params_panel(strategy):
        label_style = {"fontWeight": "600", "marginBottom": "6px"}

        if strategy == "kelly":
            return html.Div([
                html.Div("Parameters", style={"fontWeight": 700, "marginBottom": "10px"}),
                html.Div([html.Div("Initial capital", style=label_style),
                          dcc.Input(id={"type": "bt_param", "name": "capital"}, type="number", value=1000,
                                    style={"width": "100%", "padding": "8px"})]),
                html.Div([html.Div("Kelly multiplier", style=label_style),
                          dcc.Slider(id={"type": "bt_param", "name": "kelly_mult"},
                                     min=0.1, max=1.0, step=0.1, value=0.5)]),
                html.Div([html.Div("Max stake %", style=label_style),
                          dcc.Slider(id={"type": "bt_param", "name": "max_stake"},
                                     min=0.05, max=0.3, step=0.05, value=0.15)]),
            ])
        elif strategy == "top_player":
            return html.Div([
                html.Div("Parameters", style={"fontWeight": 700, "marginBottom": "10px"}),
                html.Div([html.Div("Initial capital", style=label_style),
                          dcc.Input(id={"type": "bt_param", "name": "capital"}, type="number", value=1000,
                                    style={"width": "100%", "padding": "8px"})]),
                html.Div([html.Div("Top N", style=label_style),
                          dcc.Slider(id={"type": "bt_param", "name": "top_n"},
                                     min=5, max=20, step=1, value=8)]),
                html.Div([html.Div("Kelly multiplier", style=label_style),
                          dcc.Slider(id={"type": "bt_param", "name": "kelly_mult"},
                                     min=0.1, max=1.0, step=0.1, value=0.5)]),
                html.Div([html.Div("Max stake %", style=label_style),
                          dcc.Slider(id={"type": "bt_param", "name": "max_stake"},
                                     min=0.05, max=0.3, step=0.05, value=0.15)]),
            ])
        else:
            return html.Div([
                html.Div("Parameters", style={"fontWeight": 700, "marginBottom": "10px"}),
                html.Div([html.Div("Initial capital", style=label_style),
                          dcc.Input(id={"type": "bt_param", "name": "capital"}, type="number", value=1000,
                                    style={"width": "100%", "padding": "8px"})]),
                html.Div([html.Div("Bet amount", style=label_style),
                          dcc.Input(id={"type": "bt_param", "name": "bet_amount"}, type="number", value=100,
                                    style={"width": "100%", "padding": "8px"})]),
                html.Div([html.Div("Edge threshold", style=label_style),
                          dcc.Slider(id={"type": "bt_param", "name": "threshold"},
                                     min=0.01, max=0.2, step=0.01, value=0.05)]),
            ])

    @app.callback(
        Output("bt_params_store", "data"),
        Input({"type": "bt_param", "name": ALL}, "value"),
        State("bt_strategy", "value"),
    )
    def bt_save_params(values, strategy):
        ctx = callback_context
        if not ctx.triggered:
            return {}

        # ctx.inputs_list[0] is list of dicts for pattern-matching IDs
        names = [item["id"]["name"] for item in ctx.inputs_list[0]]
        params = dict(zip(names, values))
        params["strategy"] = strategy
        return params

    @app.callback(
        Output("bt_kpi_profit", "children"),
        Output("bt_kpi_roi", "children"),
        Output("bt_kpi_final", "children"),
        Output("bt_kpi_dd", "children"),
        Output("bt_equity", "figure"),
        Output("bt_profit_by_type", "figure"),        # NEW
        Output("bt_roi_by_rankdiff", "figure"),       # NEW
        Output("bt_pnl_by_prob_bucket", "figure"),    # NEW
        Output("bt_stats_table", "children"),
        Input("bt_run", "n_clicks"),
        State("bt_params_store", "data"),
        State("bt_strategy", "value"),
        prevent_initial_call=True,
    )

    def bt_run(n_clicks, params, strategy):
        empty_fig = go.Figure().update_layout(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))

        if not params:
            return "—", "—", "—", "—", empty_fig, empty_fig, empty_fig, empty_fig, html.Div()

        df = load_backtest_df_func()
        capital = float(params.get("capital", 1000))

        # run
        if strategy == "kelly":
            bt = TwoSidedKellyBacktester(
                initial_capital=capital,
                kelly_multiplier=float(params.get("kelly_mult", 0.5)),
                max_stake_pct=float(params.get("max_stake", 0.15)),
            )
            color = "#8e44ad"
            title = "Two-Sided Kelly"
        elif strategy == "top_player":
            bt = TopPlayerKellyBacktester(
                top_n=int(params.get("top_n", 8)),
                initial_capital=capital,
                kelly_multiplier=float(params.get("kelly_mult", 0.5)),
                max_stake_pct=float(params.get("max_stake", 0.15)),
            )
            color = "#3498db"
            title = "Top Player Kelly"
        else:
            bt = TwoSidedBacktester(
                initial_capital=capital,
                bet_amount=float(params.get("bet_amount", 100)),
                threshold=float(params.get("threshold", 0.05)),
            )
            color = "#e74c3c"
            title = "Two-Sided Simple"

        bt.run(df)

        # --- NEW: analysis figs from trade log ---
        t = trades_df(bt)  # requires bt.trades to be filled in your backtester

        stats = html.Div([
            html.Div(f"Rows in df: {len(df):,}", style={"opacity": 0.8}),
            html.Div(f"Trades logged: {len(t):,}" if t is not None else "Trades logged: 0", style={"opacity": 0.8}),
            html.Div(f"Trade columns: {', '.join(list(t.columns)[:12])}" if t is not None and not t.empty else "Trade columns: (none)",
                    style={"opacity": 0.8}),
        ])
        fig_type = fig_profit_by(t, col="tournament_level", title="Profit by Tournament Level")
        fig_rank = fig_roi_by_rankdiff(t, title="ROI by Rank Difference Bucket")
        fig_prob = fig_pnl_by_prob_bucket(t, title="ROI by Predicted Probability Bucket")

        # KPIs
        profit = bt.current_capital - bt.initial_capital
        roi = (profit / bt.initial_capital) * 100.0

        peak = bt.initial_capital
        max_dd = 0.0
        for x in bt.capital_history:
            peak = max(peak, x)
            dd = (peak - x) / peak
            max_dd = max(max_dd, dd)

        k_profit = f"${profit:,.2f}"
        k_roi = f"{roi:.2f}%"
        k_final = f"${bt.current_capital:,.2f}"
        k_dd = f"{max_dd*100:.2f}%"

        # Equity curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=bt.capital_history, mode="lines", name="Equity",
            line=dict(color=color, width=3)
        ))
        fig.add_hline(y=bt.initial_capital, line_dash="dash",
                    line_color="red", annotation_text="Initial")
        fig.update_layout(
            title=dict(text=f"Equity Curve — {title}", x=0.5),
            template="plotly_white",
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
        )

        stats = html.Div([
            html.Div(f"Matches: {len(df):,}", style={"opacity": 0.8}),
            html.Div(f"Bets placed: {0 if t is None else len(t):,}", style={"opacity": 0.8}),
        ])

        # ✅ IMPORTANT: return order must match Outputs order
        return k_profit, k_roi, k_final, k_dd, fig, fig_type, fig_rank, fig_prob, stats