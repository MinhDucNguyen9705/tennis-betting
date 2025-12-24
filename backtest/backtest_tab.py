# tabs/backtest_tab.py
from dash import dcc, html, Input, Output, State, callback_context, ALL, dash_table
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from datetime import datetime, date

from .backtest_utils import (
    TwoSidedKellyBacktester, FlatKellyBacktester, TopPlayerKellyBacktester,
    FixedStakeBacktester, OddsOnlyBacktester, trades_df, fig_profit_by, fig_roi_by_rankdiff, 
    fig_pnl_by_prob_bucket, fig_bet_winloss_pct
)
from .backtest_pipeline import (
    validate_no_leakage, run_full_pipeline, train_model_only, run_backtest_only,
    TRAINING_YEARS_AVAILABLE, BACKTEST_YEARS_AVAILABLE,
    load_pretrained_model, has_pretrained_model, list_pretrained_models
)


def read_output_model():
    """Legacy function - reads pre-generated test_df.csv"""
    test_df = pd.read_csv('test_df.csv')
    return test_df


# Module-level cache for trained models (sklearn models can't be JSON-serialized)
_model_cache = {}


def make_backtest_layout():
    """Create the complete backtest tab layout with training configuration."""
    
    # Year options for training
    train_year_options = [{"label": str(y), "value": y} for y in TRAINING_YEARS_AVAILABLE]
    
    return dbc.Container(fluid=True, style={"padding": "18px"}, children=[
        # Stores
        dcc.Store(id="bt_params_store"),
        dcc.Store(id="bt_model_store", storage_type="memory"),  # Store trained model bundle (not serializable to JSON)
        dcc.Store(id="bt_predictions_store"),  # Store predictions df
        dcc.Store(id="bt_backtest_config_store"),  # Store current backtest date config
        
        # =====================================================
        # SECTION 1: Training & Backtest Configuration
        # =====================================================
        dbc.Row([
            # Training Configuration
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div([
                    html.Span("📊 ", style={"fontSize": "20px"}),
                    html.Span("Training Configuration", style={"fontSize": "18px", "fontWeight": "600"}),
                ], style={"marginBottom": "15px"}),
                
                # Model Type Selection
                dbc.Row([
                    dbc.Col([
                        html.Label("Model Type", style={"fontWeight": "600", "marginBottom": "6px"}),
                        dcc.Dropdown(
                            id="bt_model_type",
                            options=[
                                {"label": "HistGradientBoosting", "value": "hist_gradient_boosting"},
                                {"label": "RandomForest", "value": "random_forest"},
                                {"label": "Logistic Regression", "value": "logistic_regression"},
                                {"label": "SVM", "value": "svm"},
                                {"label": "CatBoost", "value": "catboost"},
                            ],
                            value="random_forest",
                            clearable=False,
                            style={"width": "100%"}
                        ),
                    ]),
                ], style={"marginBottom": "12px"}),
                
                # Pre-trained Model Toggle
                dbc.Row([
                    dbc.Col([
                        dbc.Checklist(
                            id="bt_use_pretrained",
                            options=[{"label": " Use Pre-trained Model (2021-2023)", "value": "pretrained"}],
                            value=[],
                            inline=True,
                            style={"marginBottom": "8px"}
                        ),
                        html.Div(
                            id="bt_pretrained_status",
                            style={"fontSize": "11px", "color": "#888"}
                        ),
                    ]),
                ], style={"marginBottom": "8px"}),
                
                # Training Years (hidden when using pre-trained)
                html.Div(id="bt_training_years_container", children=[
                    dbc.Row([
                        dbc.Col([
                            html.Label("Training Years", style={"fontWeight": "600", "marginBottom": "6px"}),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("From:", style={"fontSize": "12px"}),
                                    dcc.Dropdown(
                                        id="bt_train_year_start",
                                        options=train_year_options,
                                        value=2022,
                                        clearable=False,
                                        style={"width": "100%"}
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("To:", style={"fontSize": "12px"}),
                                    dcc.Dropdown(
                                        id="bt_train_year_end",
                                        options=train_year_options,
                                        value=2023,
                                        clearable=False,
                                        style={"width": "100%"}
                                    ),
                                ], width=6),
                            ]),
                            html.Div(
                                "Available: 2000-2024",
                                style={"fontSize": "11px", "color": "#888", "marginTop": "4px"}
                            ),
                        ]),
                    ]),
                ]),
            ])), width=4),
            
            # Backtest Configuration
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div([
                    html.Span("📈 ", style={"fontSize": "20px"}),
                    html.Span("Backtest Configuration", style={"fontSize": "18px", "fontWeight": "600"}),
                ], style={"marginBottom": "15px"}),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Start Date", style={"fontWeight": "600", "marginBottom": "6px"}),
                        dcc.DatePickerSingle(
                            id="bt_backtest_start",
                            min_date_allowed=date(2010, 1, 1),
                            max_date_allowed=date(2024, 12, 31),
                            initial_visible_month=date(2021, 1, 1),
                            date=date(2024, 1, 1),
                            display_format="YYYY-MM-DD",
                            style={"width": "100%"}
                        ),
                    ], width=6),
                    dbc.Col([
                        html.Label("End Date", style={"fontWeight": "600", "marginBottom": "6px"}),
                        dcc.DatePickerSingle(
                            id="bt_backtest_end",
                            min_date_allowed=date(2010, 1, 1),
                            max_date_allowed=date(2024, 12, 31),
                            initial_visible_month=date(2024, 12, 1),
                            date=date(2024, 1, 9),
                            display_format="YYYY-MM-DD",
                            style={"width": "100%"}
                        ),
                    ], width=6),
                ]),
                html.Div(
                    "Available: 2010-2024 (with odds data)",
                    style={"fontSize": "11px", "color": "#888", "marginTop": "4px"}
                ),
            ])), width=4),
            # Training & Backtest Controls
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div([
                    html.Span("🔒 ", style={"fontSize": "20px"}),
                    html.Span("Data Leakage Check", style={"fontSize": "18px", "fontWeight": "600"}),
                ], style={"marginBottom": "10px"}),
                
                html.Div(id="bt_validation_status", children=[
                    html.Div("Configure training and backtest ranges", 
                             style={"color": "#888", "fontStyle": "italic"})
                ]),
                
                html.Div(style={"height": "8px"}),
                
                # Training button
                html.Button(
                    "Train Model",
                    id="bt_train_run",
                    n_clicks=0,
                    style={
                        "width": "100%",
                        "padding": "12px",
                        "borderRadius": "8px",
                        "border": "none",
                        "backgroundColor": "#3498db",
                        "color": "white",
                        "fontWeight": "700",
                        "cursor": "pointer",
                        "fontSize": "14px"
                    }
                ),
                
                # Loading indicator wraps the status (which is the output that changes)
                dcc.Loading(
                    id="bt_train_loading",
                    type="default",
                    color="#3498db",
                    children=[
                        html.Div(id="bt_training_status", style={"marginTop": "8px", "fontSize": "11px", "minHeight": "20px"}),
                    ]
                ),
                
                html.Div(
                    "Train model once, then adjust backtest dates & strategy freely",
                    style={"fontSize": "10px", "color": "#888", "marginTop": "4px", "textAlign": "center"}
                ),
            ])), width=4),
        ], className="g-3"),
        
        html.Div(style={"height": "14px"}),
        
        # =====================================================
        # SECTION 2: KPIs and Equity Curve
        # =====================================================
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.Div("Profit", style={"fontSize": "12px", "opacity": 0.8}),
                        html.Div(id="bt_kpi_profit", children="—", style={"fontSize": "26px", "fontWeight": 700}),
                    ])), width=3),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.Div("ROI", style={"fontSize": "12px", "opacity": 0.8}),
                        html.Div(id="bt_kpi_roi", children="—", style={"fontSize": "26px", "fontWeight": 700}),
                    ])), width=3),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.Div("Final capital", style={"fontSize": "12px", "opacity": 0.8}),
                        html.Div(id="bt_kpi_final", children="—", style={"fontSize": "26px", "fontWeight": 700}),
                    ])), width=3),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.Div("Max DD", style={"fontSize": "12px", "opacity": 0.8}),
                        html.Div(id="bt_kpi_dd", children="—", style={"fontSize": "26px", "fontWeight": 700}),
                    ])), width=3),
                ], className="g-2"),
                html.Div(style={"height": "12px"}),
                dcc.Loading(dcc.Graph(id="bt_equity", config={"displayModeBar": False}, style={"height": "420px"})),
                html.Div(id="bt_stats_table", style={"marginTop": "10px"}),
            ])), width=8),

            # Strategy Selection Panel
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Backtest Strategy", style={"fontSize": "18px", "fontWeight": "600", "marginBottom": "10px"}),
                dcc.Dropdown(
                    id="bt_strategy",
                    options=[
                        {"label": "Fractional Kelly", "value": "kelly"},
                        {"label": "Flat Kelly", "value": "flat_kelly"},
                        {"label": "Top Player Kelly", "value": "kelly_top_player"},
                        {"label": "Flat Staking", "value": "fixed_stake"},
                        {"label": "Odds Only", "value": "odds_only"},
                    ],
                    value="kelly",
                    clearable=False,
                ),
                html.Div(id="bt_params_panel", style={"marginTop": "12px"}),
                dcc.Loading(
                    id="bt_backtest_loading",
                    type="circle",
                    color="#27ae60",
                    children=[
                        html.Button(
                            "📈 Run Backtest",
                            id="bt_run",
                            n_clicks=0,
                            style={
                                "width": "100%",
                                "marginTop": "12px",
                                "padding": "12px",
                                "borderRadius": "8px",
                                "border": "none",
                                "backgroundColor": "#27ae60",
                                "color": "white",
                                "fontWeight": "700"
                            }
                        ),
                    ]
                ),
                html.Div(
                    "Uses trained model to generate predictions & run strategy",
                    style={"fontSize": "11px", "color": "#888", "marginTop": "6px", "textAlign": "center"}
                ),
            ])), width=4),
        ], className="g-3"),

        html.Div(style={"height": "14px"}),
        
        # =====================================================
        # SECTION 3: Analysis Charts
        # =====================================================
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Win/Loss Distribution", style={"fontWeight": 600, "marginBottom": 8}),
                dcc.Graph(id="bt_winloss_pct", config={"displayModeBar": False}, style={"height": "320px"})
            ])), width=6),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("ROI by Rank Difference", style={"fontWeight": 600, "marginBottom": 8}),
                dcc.Graph(id="bt_roi_by_rankdiff", config={"displayModeBar": False}, style={"height": "320px"})
            ])), width=6),
        ], className="g-3"),

        html.Div(style={"height": "14px"}),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Profit by Tournament Type", style={"fontWeight": 600, "marginBottom": 8}),
                dcc.Graph(id="bt_profit_by_type", config={"displayModeBar": False}, style={"height": "320px"})
            ])), width=6),
            
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("ROI by Probability Bucket", style={"fontWeight": 600, "marginBottom": 8}),
                dcc.Graph(id="bt_pnl_by_prob_bucket", config={"displayModeBar": False}, style={"height": "320px"})
            ])), width=6),
        ], className="g-3"),

        html.Div(style={"height": "14px"}),

        # =====================================================
        # SECTION 4: Bet History Table
        # =====================================================
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div([
                    html.Span("📋", style={"fontSize": "18px"}),
                    html.Span("Bet History", style={"fontWeight": 600, "fontSize": "16px"}),
                    html.Span(id="bt_bet_count", style={"marginLeft": "10px", "fontSize": "12px", "color": "#888"}),
                ], style={"marginBottom": 12}),
                dcc.Loading(
                    dash_table.DataTable(
                        id="bt_bet_history",
                        page_size=15,
                        page_action="native",
                        sort_action="native",
                        filter_action="native",
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": "#1e1e2f",
                            "color": "white",
                            "fontWeight": "bold",
                            "textAlign": "center",
                            "padding": "10px",
                        },
                        style_cell={
                            "fontFamily": "'Segoe UI', Arial, sans-serif",
                            "fontSize": "12px",
                            "padding": "8px",
                            "textAlign": "center",
                        },
                        style_data_conditional=[
                            {
                                "if": {"filter_query": "{PnL} > 0"},
                                "backgroundColor": "rgba(39, 174, 96, 0.15)",
                                "color": "#27ae60",
                            },
                            {
                                "if": {"filter_query": "{PnL} < 0"},
                                "backgroundColor": "rgba(231, 76, 60, 0.15)",
                                "color": "#e74c3c",
                            },
                            {
                                "if": {"column_id": "Result", "filter_query": "{Result} = 'WIN'"},
                                "fontWeight": "bold",
                                "color": "#27ae60",
                            },
                            {
                                "if": {"column_id": "Result", "filter_query": "{Result} = 'LOSS'"},
                                "fontWeight": "bold",
                                "color": "#e74c3c",
                            },
                        ],
                    )
                ),
            ])), width=12),
        ], className="g-3"),
    ])


def register_backtest_callbacks(app, load_backtest_df_func=None):
    """
    Register all callbacks for the backtest tab.
    
    Args:
        app: Dash app instance
        load_backtest_df_func: Optional legacy function to load backtest dataframe.
                              If None, will use the new train & backtest pipeline.
    """

    # =========================================================
    # Callback 1: Validate data leakage on config change
    # =========================================================
    @app.callback(
        Output("bt_validation_status", "children"),
        Input("bt_train_year_start", "value"),
        Input("bt_train_year_end", "value"),
        Input("bt_backtest_start", "date"),
    )
    def validate_config(train_start, train_end, backtest_start):
        if not all([train_start, train_end, backtest_start]):
            return html.Div("Please configure all parameters", style={"color": "#888"})
        
        train_years = list(range(train_start, train_end + 1))
        is_valid, message = validate_no_leakage(train_years, backtest_start)
        
        if is_valid:
            return html.Div([
                html.Span("✅ ", style={"fontSize": "16px"}),
                html.Span(message.replace("✅ ", ""), style={"color": "#27ae60", "fontWeight": "600"})
            ])
        else:
            return html.Div([
                html.Span("❌ ", style={"fontSize": "16px"}),
                html.Span(message, style={"color": "#e74c3c", "fontWeight": "600"})
            ])

    # =========================================================
    # Callback 1b: Toggle training years visibility based on pre-trained checkbox
    # =========================================================
    @app.callback(
        Output("bt_training_years_container", "style"),
        Output("bt_pretrained_status", "children"),
        Output("bt_train_run", "disabled"),
        Input("bt_use_pretrained", "value"),
        Input("bt_model_type", "value"),
    )
    def toggle_pretrained_mode(use_pretrained, model_type):
        if "pretrained" in (use_pretrained or []):
            # Check if pretrained model exists
            if has_pretrained_model(model_type):
                status = html.Div([
                    html.Span("✅ ", style={"color": "#27ae60"}),
                    html.Span("Pre-trained model available", style={"color": "#27ae60"})
                ])
            else:
                status = html.Div([
                    html.Span("⚠️ ", style={"color": "#f39c12"}),
                    html.Span("No pre-trained model found - will need to train", style={"color": "#f39c12"})
                ])
            return {"display": "none"}, status, True
        else:
            return {"display": "block"}, "", False

    # =========================================================
    # Callback 2: Strategy parameters panel
    # =========================================================
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
                          html.Div(dcc.Slider(id={"type": "bt_param", "name": "kelly_mult"},
                                     min=0.1, max=0.5, step=0.05, value=0.25,
                                     marks={0.1: "10%", 0.2: "20%", 0.3: "30%", 0.4: "40%", 0.5: "50%"}),
                                   style={"padding": "0 10px", "marginBottom": "15px"})]),
                html.Div([html.Div("Max stake %", style=label_style),
                          html.Div(dcc.Slider(id={"type": "bt_param", "name": "max_stake"},
                                     min=0.02, max=0.15, step=0.01, value=0.05,
                                     marks={0.02: "2%", 0.05: "5%", 0.08: "8%", 0.10: "10%", 0.12: "12%", 0.15: "15%"}),
                                   style={"padding": "0 10px", "marginBottom": "15px"})]),
            ])
        elif strategy == "flat_kelly":
            return html.Div([
                html.Div("Parameters", style={"fontWeight": 700, "marginBottom": "10px"}),
                html.Div("Uses initial capital for bet sizing (prevents shrinking bets during drawdowns)",
                         style={"fontSize": "11px", "color": "#888", "marginBottom": "10px", "fontStyle": "italic"}),
                html.Div([html.Div("Initial capital", style=label_style),
                          dcc.Input(id={"type": "bt_param", "name": "capital"}, type="number", value=1000,
                                    style={"width": "100%", "padding": "8px"})]),
                html.Div([html.Div("Kelly multiplier", style=label_style),
                          html.Div(dcc.Slider(id={"type": "bt_param", "name": "kelly_mult"},
                                     min=0.1, max=0.5, step=0.05, value=0.25,
                                     marks={0.1: "10%", 0.2: "20%", 0.3: "30%", 0.4: "40%", 0.5: "50%"}),
                                   style={"padding": "0 10px", "marginBottom": "15px"})]),
                html.Div([html.Div("Max stake %", style=label_style),
                          html.Div(dcc.Slider(id={"type": "bt_param", "name": "max_stake"},
                                     min=0.02, max=0.15, step=0.01, value=0.08,
                                     marks={0.02: "2%", 0.05: "5%", 0.08: "8%", 0.10: "10%", 0.12: "12%", 0.15: "15%"}),
                                   style={"padding": "0 10px", "marginBottom": "15px"})]),
            ])
        elif strategy == "kelly_top_player":
            return html.Div([
                html.Div("Parameters", style={"fontWeight": 700, "marginBottom": "10px"}),
                html.Div([html.Div("Initial capital", style=label_style),
                          dcc.Input(id={"type": "bt_param", "name": "capital"}, type="number", value=1000,
                                    style={"width": "100%", "padding": "8px"})]),
                html.Div([html.Div("Top N", style=label_style),
                          html.Div(dcc.Slider(id={"type": "bt_param", "name": "top_n"},
                                     min=5, max=20, step=1, value=8,
                                     marks={5: "5", 8: "8", 10: "10", 15: "15", 20: "20"}),
                                   style={"padding": "0 10px", "marginBottom": "15px"})]),
                html.Div([html.Div("Kelly multiplier", style=label_style),
                          html.Div(dcc.Slider(id={"type": "bt_param", "name": "kelly_mult"},
                                     min=0.1, max=0.5, step=0.05, value=0.25,
                                     marks={0.1: "10%", 0.2: "20%", 0.3: "30%", 0.4: "40%", 0.5: "50%"}),
                                   style={"padding": "0 10px", "marginBottom": "15px"})]),
                html.Div([html.Div("Max stake %", style=label_style),
                          html.Div(dcc.Slider(id={"type": "bt_param", "name": "max_stake"},
                                     min=0.02, max=0.15, step=0.01, value=0.05,
                                     marks={0.02: "2%", 0.05: "5%", 0.08: "8%", 0.10: "10%", 0.12: "12%", 0.15: "15%"}),
                                   style={"padding": "0 10px", "marginBottom": "15px"})]),
            ])
        elif strategy == "fixed_stake":
            return html.Div([
                html.Div("Parameters", style={"fontWeight": 700, "marginBottom": "10px"}),
                html.Div([html.Div("Initial capital", style=label_style),
                          dcc.Input(id={"type": "bt_param", "name": "capital"}, type="number", value=1000,
                                    style={"width": "100%", "padding": "8px"})]),
                html.Div([html.Div("Stake amount per bet", style=label_style),
                          dcc.Input(id={"type": "bt_param", "name": "stake_amount"}, type="number", value=100,
                                    style={"width": "100%", "padding": "8px"})]),
                html.Div([html.Div("Min probability to bet", style=label_style),
                          html.Div(dcc.Slider(id={"type": "bt_param", "name": "min_prob"},
                                     min=0.5, max=0.7, step=0.05, value=0.5,
                                     marks={0.5: "50%", 0.6: "60%", 0.7: "70%"}),
                                   style={"padding": "0 10px", "marginBottom": "15px"})]),
            ])
        elif strategy == "odds_only":
            return html.Div([
                html.Div("Parameters", style={"fontWeight": 700, "marginBottom": "10px"}),
                html.Div([html.Div("Initial capital", style=label_style),
                          dcc.Input(id={"type": "bt_param", "name": "capital"}, type="number", value=1000,
                                    style={"width": "100%", "padding": "8px"})]),
                html.Div([html.Div("Stake amount per bet", style=label_style),
                          dcc.Input(id={"type": "bt_param", "name": "stake_amount"}, type="number", value=100,
                                    style={"width": "100%", "padding": "8px"})]),
                html.Div([html.Div("Bet on", style=label_style),
                          dcc.Dropdown(
                              id={"type": "bt_param", "name": "bet_on"},
                              options=[
                                  {"label": "Favorite (lower odds)", "value": "favorite"},
                                  {"label": "Underdog (higher odds)", "value": "underdog"},
                              ],
                              value="favorite",
                              clearable=False,
                              style={"width": "100%"}
                          )]),
                html.Div([html.Div("Min odds", style=label_style),
                          html.Div(dcc.Slider(id={"type": "bt_param", "name": "min_odds"},
                                     min=1.1, max=2.0, step=0.1, value=1.2,
                                     marks={1.1: "1.1", 1.5: "1.5", 2.0: "2.0"}),
                                   style={"padding": "0 10px", "marginBottom": "15px"})]),
                html.Div([html.Div("Max odds", style=label_style),
                          html.Div(dcc.Slider(id={"type": "bt_param", "name": "max_odds"},
                                     min=1.5, max=5.0, step=0.5, value=3.0,
                                     marks={1.5: "1.5", 3.0: "3.0", 5.0: "5.0"}),
                                   style={"padding": "0 10px", "marginBottom": "15px"})]),
            ])
        else:
            return html.Div()

    # =========================================================
    # Callback 3: Store parameters
    # =========================================================
    @app.callback(
        Output("bt_params_store", "data"),
        Input({"type": "bt_param", "name": ALL}, "value"),
        State("bt_strategy", "value"),
    )
    def bt_save_params(values, strategy):
        ctx = callback_context
        if not ctx.triggered:
            return {}

        names = [item["id"]["name"] for item in ctx.inputs_list[0]]
        params = dict(zip(names, values))
        params["strategy"] = strategy
        return params

    # =========================================================
    # Callback 4: Train Model Only (stores model bundle)
    # =========================================================
    # Uses module-level _model_cache since dcc.Store can't serialize sklearn models
    
    @app.callback(
        Output("bt_model_store", "data"),
        Output("bt_training_status", "children"),
        Input("bt_train_run", "n_clicks"),
        State("bt_train_year_start", "value"),
        State("bt_train_year_end", "value"),
        State("bt_model_type", "value"),
        prevent_initial_call=True,
    )
    def train_model_callback(n_clicks, train_start, train_end, model_type):
        if not n_clicks:
            return None, ""
        
        try:
            train_years = list(range(train_start, train_end + 1))
            
            # Map model type to display name
            model_names = {
                'hist_gradient_boosting': 'HistGradientBoosting',
                'random_forest': 'RandomForest',
                'logistic_regression': 'Logistic Regression',
                'svm': 'SVM',
                'catboost': 'CatBoost',
            }
            model_display_name = model_names.get(model_type, model_type)
            
            # Train model only (no backtest)
            model_bundle = train_model_only(
                train_years=train_years,
                use_calibration=True,
                model_type=model_type
            )
            
            # Store model in cache (not JSON-serializable)
            cache_key = f"model_{model_type}_{train_start}_{train_end}"
            _model_cache[cache_key] = model_bundle
            
            # Return metadata for the store
            model_meta = {
                "cache_key": cache_key,
                "train_years": train_years,
                "max_train_year": max(train_years),
                "trained_at": model_bundle.get("trained_at", ""),
                "features_count": len(model_bundle.get("features", [])),
                "model_type": model_type,
            }
            
            status = html.Div([
                html.Span("✅ ", style={"fontSize": "12px"}),
                html.Span(f"{model_display_name} trained ({train_start}-{train_end}). Ready to run backtest!",
                         style={"color": "#27ae60", "fontSize": "11px"})
            ])
            
            return model_meta, status
            
        except Exception as e:
            return None, html.Div(f"❌ {str(e)}", style={"color": "#e74c3c", "fontSize": "11px"})

    # =========================================================
    # Callback 5: Run Backtest (generates predictions + runs strategy)
    # =========================================================
    @app.callback(
        Output("bt_kpi_profit", "children"),
        Output("bt_kpi_roi", "children"),
        Output("bt_kpi_final", "children"),
        Output("bt_kpi_dd", "children"),
        Output("bt_equity", "figure"),
        Output("bt_profit_by_type", "figure"), 
        Output("bt_roi_by_rankdiff", "figure"), 
        Output("bt_pnl_by_prob_bucket", "figure"),  
        Output("bt_winloss_pct", "figure"),
        Output("bt_stats_table", "children"),
        Output("bt_bet_history", "data"),
        Output("bt_bet_history", "columns"),
        Output("bt_bet_count", "children"),
        Output("bt_predictions_store", "data"),
        Input("bt_run", "n_clicks"),
        State("bt_model_store", "data"),
        State("bt_backtest_start", "date"),
        State("bt_backtest_end", "date"),
        State("bt_params_store", "data"),
        State("bt_strategy", "value"),
        State("bt_use_pretrained", "value"),
        State("bt_model_type", "value"),
        prevent_initial_call=True,
    )
    def bt_run(n_clicks, model_meta, backtest_start, backtest_end, params, strategy, use_pretrained, model_type):
        empty_fig = go.Figure().update_layout(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
        empty_return = ("—", "—", "—", "—", empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, html.Div(), [], [], "", None)

        if not n_clicks:
            return empty_return
        
        # Check if using pre-trained model
        using_pretrained = "pretrained" in (use_pretrained or [])
        
        if using_pretrained:
            # Load pre-trained model
            model_bundle = load_pretrained_model(model_type)
            if model_bundle is None:
                empty_return_with_msg = list(empty_return)
                empty_return_with_msg[9] = html.Div(
                    f"⚠️ No pre-trained model found for {model_type}. Please train first or run train_pretrained_models.py", 
                    style={"color": "#f39c12"}
                )
                return tuple(empty_return_with_msg)
            
            # Store in cache for potential reuse
            cache_key = f"pretrained_{model_type}"
            _model_cache[cache_key] = model_bundle
        else:
            # Check if model is trained
            if not model_meta or "cache_key" not in model_meta:
                empty_return_with_msg = list(empty_return)
                empty_return_with_msg[9] = html.Div("⚠️ Please train model first!", style={"color": "#f39c12"})
                return tuple(empty_return_with_msg)
            
            # Retrieve model from cache
            cache_key = model_meta["cache_key"]
            if cache_key not in _model_cache:
                empty_return_with_msg = list(empty_return)
                empty_return_with_msg[9] = html.Div("⚠️ Model expired. Please retrain.", style={"color": "#f39c12"})
                return tuple(empty_return_with_msg)
            
            model_bundle = _model_cache[cache_key]
        
        # Validate no data leakage
        max_train_year = model_bundle.get("max_train_year")
        if max_train_year:
            is_valid, message = validate_no_leakage([max_train_year], backtest_start)
            if not is_valid:
                empty_return_with_msg = list(empty_return)
                empty_return_with_msg[9] = html.Div(f"❌ {message}", style={"color": "#e74c3c"})
                return tuple(empty_return_with_msg)
        
        try:
            # Run inference to get predictions
            predictions_df = run_backtest_only(
                model_bundle=model_bundle,
                backtest_start_date=backtest_start,
                backtest_end_date=backtest_end,
            )
            
            # Store predictions as JSON
            predictions_json = predictions_df.to_json(date_format='iso', orient='split')
            df = predictions_df
            
        except Exception as e:
            empty_return_with_msg = list(empty_return)
            empty_return_with_msg[9] = html.Div(f"❌ Inference error: {str(e)}", style={"color": "#e74c3c"})
            return tuple(empty_return_with_msg)
        
        if not params:
            params = {"capital": 1000}
        
        capital = float(params.get("capital", 1000))

        # Run selected strategy
        if strategy == "kelly":
            bt = TwoSidedKellyBacktester(
                initial_capital=capital,
                kelly_multiplier=float(params.get("kelly_mult", 0.25)),
                max_stake_pct=float(params.get("max_stake", 0.05)),
            )
            color = "#8e44ad"
            title = "Fractional Kelly"
        elif strategy == "flat_kelly":
            bt = FlatKellyBacktester(
                initial_capital=capital,
                kelly_multiplier=float(params.get("kelly_mult", 0.25)),
                max_stake_pct=float(params.get("max_stake", 0.08)),
            )
            color = "#27ae60"
            title = "Flat Kelly"
        elif strategy == "kelly_top_player":
            bt = TopPlayerKellyBacktester(
                top_n=int(params.get("top_n", 8)),
                initial_capital=capital,
                kelly_multiplier=float(params.get("kelly_mult", 0.25)),
                max_stake_pct=float(params.get("max_stake", 0.05)),
            )
            color = "#3498db"
            title = "Top Player Kelly"
        elif strategy == "fixed_stake":
            bt = FixedStakeBacktester(
                initial_capital=capital,
                stake_amount=float(params.get("stake_amount", 100)),
                min_prob=float(params.get("min_prob", 0.5)),
            )
            color = "#f39c12"
            title = "Flat Staking"
        elif strategy == "odds_only":
            bt = OddsOnlyBacktester(
                initial_capital=capital,
                stake_amount=float(params.get("stake_amount", 100)),
                bet_on=str(params.get("bet_on", "favorite")),
                min_odds=float(params.get("min_odds", 1.2)),
                max_odds=float(params.get("max_odds", 3.0)),
            )
            color = "#9b59b6"
            title = f"Odds Only ({params.get('bet_on', 'favorite').title()})"

        bt.run(df)

        # Analysis figures from trade log
        t = trades_df(bt)
        
        fig_type = fig_profit_by(t, col="tournament_level", title="Profit by Tournament Level")
        fig_rank = fig_roi_by_rankdiff(t, title="ROI by Rank Difference Bucket")
        fig_prob = fig_pnl_by_prob_bucket(t, title="ROI by Predicted Probability Bucket")
        fig_wl = fig_bet_winloss_pct(t)

        # KPIs
        profit = bt.current_capital - bt.initial_capital
        roi = (profit / bt.initial_capital) * 100.0

        peak = bt.initial_capital
        max_dd = 0.0
        dd_history = []
        for x in bt.capital_history:
            peak = max(peak, x)
            dd = (peak - x) / peak if peak > 0 else 0
            dd_history.append(dd)
            max_dd = max(max_dd, dd)

        # Color based on positive/negative values
        profit_color = "#27ae60" if profit >= 0 else "#e74c3c"
        roi_color = "#27ae60" if roi >= 0 else "#e74c3c"
        final_color = "#27ae60" if bt.current_capital >= bt.initial_capital else "#e74c3c"
        
        k_profit = html.Span(f"${profit:,.2f}", style={"color": profit_color})
        k_roi = html.Span(f"{roi:.2f}%", style={"color": roi_color})
        k_final = html.Span(f"${bt.current_capital:,.2f}", style={"color": final_color})
        k_dd = f"{max_dd*100:.2f}%"

        # ========================================
        # Enhanced Professional Equity Curve
        # ========================================
        fig = go.Figure()
        
        # Calculate running max (peak) for drawdown visualization
        running_peak = []
        peak_val = bt.initial_capital
        for x in bt.capital_history:
            peak_val = max(peak_val, x)
            running_peak.append(peak_val)
        
        # Add underwater/drawdown fill area
        fig.add_trace(go.Scatter(
            y=running_peak,
            mode="lines",
            name="Peak",
            line=dict(color="rgba(150, 150, 150, 0.3)", width=1, dash="dot"),
            showlegend=False,
        ))
        
        # Add equity curve with gradient fill
        fig.add_trace(go.Scatter(
            y=bt.capital_history,
            mode="lines",
            name="Equity",
            line=dict(color=color, width=2.5),
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)",
        ))
        
        # Add markers for significant drawdowns
        if len(bt.capital_history) > 0:
            fig.add_hline(
                y=bt.initial_capital,
                line_dash="dash",
                line_color="#e74c3c",
                line_width=1.5,
                annotation_text=f"Initial: ${bt.initial_capital:,.0f}",
                annotation_position="right",
                annotation_font_size=10,
                annotation_font_color="#e74c3c",
            )
            
            # Add final capital annotation
            final_color = "#27ae60" if profit > 0 else "#e74c3c"
            fig.add_annotation(
                x=len(bt.capital_history) - 1,
                y=bt.current_capital,
                text=f"${bt.current_capital:,.0f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowcolor=final_color,
                font=dict(size=11, color=final_color, weight="bold"),
                ax=30,
                ay=-30,
            )
        
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b> │ ROI: {roi:.1f}% │ Max DD: {max_dd*100:.1f}%",
                x=0.5,
                font=dict(size=14, color="#333"),
            ),
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=420,
            margin=dict(l=60, r=60, t=50, b=40),
            xaxis=dict(
                title="Trade #",
                showgrid=True,
                gridcolor="rgba(0,0,0,0.1)",
                zeroline=False,
            ),
            yaxis=dict(
                title="Capital ($)",
                showgrid=True,
                gridcolor="rgba(0,0,0,0.1)",
                zeroline=False,
                tickformat="$,.0f",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            hovermode="x unified",
        )

        stats = html.Div([
            html.Div(f"Matches analyzed: {len(df):,}", style={"opacity": 0.8}),
            html.Div(f"Bets placed: {0 if t is None else len(t):,}", style={"opacity": 0.8}),
        ])

        # ========================================
        # Prepare Bet History Table Data
        # ========================================
        bet_history_data = []
        bet_columns = []
        bet_count_text = ""
        
        if t is not None and not t.empty:
            # Format the trades for display
            history_df = t.copy()
            
            # Create formatted columns
            formatted = pd.DataFrame()
            formatted["#"] = range(1, len(history_df) + 1)
            
            if "date" in history_df.columns:
                formatted["Date"] = pd.to_datetime(history_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            
            # Add tournament name
            if "tournament" in history_df.columns:
                formatted["Tournament"] = history_df["tournament"].fillna("-")
            
            # Add player names
            if "player1" in history_df.columns:
                formatted["Player 1"] = history_df["player1"].fillna("-")
            if "player2" in history_df.columns:
                formatted["Player 2"] = history_df["player2"].fillna("-")
            
            if "bet_side" in history_df.columns:
                formatted["Bet On"] = history_df["bet_side"]
            
            if "prob" in history_df.columns:
                formatted["Prob"] = (history_df["prob"] * 100).round(1).astype(str) + "%"
            
            if "odds" in history_df.columns:
                formatted["Odds"] = history_df["odds"].round(2)
            
            if "stake" in history_df.columns:
                formatted["Stake"] = "$" + history_df["stake"].round(2).astype(str)
            
            if "pnl" in history_df.columns:
                formatted["PnL"] = history_df["pnl"].round(2)
            
            if "is_win" in history_df.columns:
                formatted["Result"] = history_df["is_win"].apply(lambda x: "WIN" if x == 1 else "LOSS")
            
            if "tournament_level" in history_df.columns:
                formatted["Level"] = history_df["tournament_level"]
            
            if "surface" in history_df.columns:
                formatted["Surface"] = history_df["surface"]
            
            # Calculate cumulative PnL for running total
            if "pnl" in history_df.columns:
                formatted["Cumul. PnL"] = history_df["pnl"].cumsum().round(2)
            
            bet_history_data = formatted.to_dict("records")
            bet_columns = [{"name": col, "id": col} for col in formatted.columns]
            
            wins = (history_df["is_win"] == 1).sum() if "is_win" in history_df.columns else 0
            losses = len(history_df) - wins
            win_rate = wins / len(history_df) * 100 if len(history_df) > 0 else 0
            bet_count_text = f"({len(history_df):,} bets │ {wins}W / {losses}L │ {win_rate:.1f}% win rate)"

        return k_profit, k_roi, k_final, k_dd, fig, fig_type, fig_rank, fig_prob, fig_wl, stats, bet_history_data, bet_columns, bet_count_text, predictions_json