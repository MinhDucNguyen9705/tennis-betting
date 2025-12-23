import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

def kpi_card(title, value_id):
    return dbc.Card(dbc.CardBody([
        html.Div(title, style={"fontSize": "14px", "opacity": 0.8}),
        html.Div(id=value_id, style={"fontSize": "28px", "fontWeight": "600", "marginTop": "6px"}),
    ]))

def make_h2h_layout(player_dim, levels, min_dt, max_dt, default_p1):
    return dbc.Container(fluid=True, style={"padding": "18px"}, children=[
        dbc.Row([
            dbc.Col(
                dbc.Card(dbc.CardBody([
                    dbc.Row([
                        dbc.Col(kpi_card("No. Match", "kpi_matches"), width=4),
                        dbc.Col(kpi_card("Victory rate", "kpi_winrate"), width=4),
                        dbc.Col(kpi_card("Pred Victory rate", "kpi_predrate"), width=4),
                    ], className="g-2"),
                ])),
                width=8
            ),
            dbc.Col(
                dbc.Card(dbc.CardBody([
                    html.Div("Filter", style={"fontSize": "18px", "fontWeight": "600", "marginBottom": "10px"}),
                    dbc.Row([
                        dbc.Col([
                            html.Div("Player 1", style={"fontSize": "12px", "opacity": 0.8}),
                            dcc.Dropdown(
                                id="player1",
                                options=[{"label": f"{r.display_name} (ID: {r.id})", "value": r.id}
                                         for r in player_dim.itertuples(index=False)],
                                value=default_p1,
                                clearable=False
                            )
                        ], width=6, style={"minWidth": 0}),
                        dbc.Col([
                            html.Div("Player 2", style={"fontSize": "12px", "opacity": 0.8}),
                            dcc.Dropdown(id="player2",
                                         options=[{"label": "(All)", "value": "(All)"}],
                                         value="(All)", clearable=False)
                        ], width=6, style={"minWidth": 0}),
                    ], className="g-2", style={"marginBottom": "10px"}),

                    dbc.Row([
                        dbc.Col([
                            html.Div("Time Range", style={"fontSize": "12px", "opacity": 0.8}),
                            html.Div(dcc.DatePickerRange(
                                id="date_range",
                                min_date_allowed=min_dt, max_date_allowed=max_dt,
                                start_date=min_dt, end_date=max_dt,
                                display_format="YYYY-MM-DD",
                                number_of_months_shown=1,
                            ), style={"width": "100%"})
                        ], width=8, style={"minWidth": 0}),
                        dbc.Col([
                            html.Div("Tour. Level", style={"fontSize": "12px", "opacity": 0.8}),
                            dcc.Dropdown(
                                id="tour_level",
                                options=[{"label": "(All)", "value": "(All)"}] + [{"label": lv, "value": lv} for lv in levels],
                                value="(All)", clearable=False
                            )
                        ], width=4, style={"minWidth": 0}),
                    ], className="g-2"),
                ]), style={"height": "auto"}),
                width=4
            ),
        ], className="g-3"),

        html.Div(style={"height": "14px"}),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Win/Loss by Surface (stacked)", style={"fontWeight": "600", "marginBottom": "8px"}),
                dcc.Graph(id="surface_bar", config={"displayModeBar": False}, style={"height": "340px"})
            ]), style={"height": "420px"}), width=8),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Skill Radar (Player 1 vs Player 2)", style={"fontWeight": "600", "marginBottom": "8px"}),
                dcc.Graph(id="skill_radar", config={"displayModeBar": False}, style={"height": "340px"})
            ]), style={"height": "420px"}), width=4),
        ], className="g-3"),

        html.Div(style={"height": "14px"}),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Detail table", style={"fontWeight": "600", "marginBottom": "8px"}),
                dash_table.DataTable(
                    id="detail_table",
                    page_size=12,
                    style_table={"overflowX": "auto"},
                    style_cell={"fontFamily": "Arial", "fontSize": "12px", "padding": "6px"},
                    style_header={"fontWeight": "600"},
                    sort_action="native",
                    filter_action="native",
                )
            ]), style={"height": "420px"}), width=12)
        ])
    ])

def make_profile_layout(player_dim, levels, min_dt, max_dt, default_p1):
    profile_filter_card = dbc.Card(dbc.CardBody([
        html.Div("Player Profile Filters", style={"fontSize": "18px", "fontWeight": "600", "marginBottom": "10px"}),

        dbc.Row([
            dbc.Col([
                html.Div("Player", style={"fontSize": "12px", "opacity": 0.8}),
                dcc.Dropdown(
                    id="p2_player",
                    options=[{"label": r.display_name, "value": r.id} for r in player_dim.itertuples(index=False)],
                    value=default_p1, clearable=False, searchable=True
                ),
                html.Div(id="p2_player_hint", style={"fontSize": "12px", "opacity": 0.7, "marginTop": "4px"})
            ], width=6),

            dbc.Col([
                html.Div("Mode", style={"fontSize": "12px", "opacity": 0.8}),
                dbc.RadioItems(
                    id="p2_mode",
                    options=[{"label": "All time", "value": "all"},
                             {"label": "Use filters below", "value": "filter"}],
                    value="all", inline=True
                ),
            ], width=6),
        ], className="g-2", style={"marginBottom": "10px"}),

        dbc.Row([
            dbc.Col([
                html.Div("Time Range", style={"fontSize": "12px", "opacity": 0.8}),
                html.Div(dcc.DatePickerRange(
                    id="p2_date_range",
                    min_date_allowed=min_dt, max_date_allowed=max_dt,
                    start_date=min_dt, end_date=max_dt,
                    display_format="YYYY-MM-DD",
                    number_of_months_shown=1,
                ), style={"width": "100%"})
            ], width=8),

            dbc.Col([
                html.Div("Tour. Level", style={"fontSize": "12px", "opacity": 0.8}),
                dcc.Dropdown(
                    id="p2_tour_level",
                    options=[{"label": "(All)", "value": "(All)"}] + [{"label": lv, "value": lv} for lv in levels],
                    value="(All)", clearable=False
                )
            ], width=4),
        ], className="g-2"),
    ]))

    top_profile_row = dbc.Row([
        dbc.Col(dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Matches (All time)", style={"fontSize": "14px", "opacity": 0.8}),
                html.Div(id="p2_kpi_matches", style={"fontSize": "28px", "fontWeight": "600", "marginTop": "6px"}),
            ])), width=12),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Win rate", style={"fontSize": "14px", "opacity": 0.8}),
                html.Div(id="p2_kpi_wr", style={"fontSize": "28px", "fontWeight": "600", "marginTop": "6px"}),
            ])), width=12),
        ], className="g-3"), width=4),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Win/Loss + Round Breakdown", style={"fontWeight": "600", "marginBottom": "8px"}),
            dcc.Graph(id="p2_win_round_donut", config={"displayModeBar": False}, style={"height": "240px"})
        ])), width=4),

        dbc.Col(dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Form win rate (last 10)", style={"fontSize": "14px", "opacity": 0.8}),
                html.Div(id="p2_kpi_form10", style={"fontSize": "28px", "fontWeight": "600", "marginTop": "6px"}),
            ])), width=12),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Aces per match", style={"fontSize": "14px", "opacity": 0.8}),
                html.Div(id="p2_kpi_aces", style={"fontSize": "28px", "fontWeight": "600", "marginTop": "6px"}),
            ])), width=12),
        ], className="g-3"), width=4),
    ], className="g-3")

    top_profile_row = dbc.Row([
        # LEFT: Matches + Form (moved here)
        dbc.Col(dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Matches (All time)", style={"fontSize": "14px", "opacity": 0.8}),
                html.Div(id="p2_kpi_matches", style={"fontSize": "28px", "fontWeight": "600", "marginTop": "6px"}),
            ])), width=12),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Win rate", style={"fontSize": "14px", "opacity": 0.8}),
                html.Div(id="p2_kpi_wr", style={"fontSize": "28px", "fontWeight": "600", "marginTop": "6px"}),
            ])), width=12),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Form win rate (last 10)", style={"fontSize": "14px", "opacity": 0.8}),
                html.Div(id="p2_kpi_form10", style={"fontSize": "28px", "fontWeight": "600", "marginTop": "6px"}),
            ])), width=12),
        ], className="g-3"), width=4),

        # MIDDLE: Donut
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Win/Loss + Round Breakdown", style={"fontWeight": "600", "marginBottom": "8px"}),
            dcc.Graph(id="p2_win_round_donut", config={"displayModeBar": False}, style={"height": "240px"})
        ])), width=4),

        # RIGHT: Aces + new metrics (Serve points + BP saved)
        dbc.Col(dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Avg aces", style={"fontSize": "14px", "opacity": 0.8}),
                html.Div(id="p2_kpi_aces", style={"fontSize": "28px", "fontWeight": "600", "marginTop": "6px"}),
            ])), width=12),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Avg serve points", style={"fontSize": "14px", "opacity": 0.8}),
                html.Div(id="p2_kpi_svpt", style={"fontSize": "28px", "fontWeight": "600", "marginTop": "6px"}),
            ])), width=12),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Avg BP saved", style={"fontSize": "14px", "opacity": 0.8}),
                html.Div(id="p2_kpi_bpSaved", style={"fontSize": "28px", "fontWeight": "600", "marginTop": "6px"}),
            ])), width=12),
        ], className="g-3"), width=4),
    ], className="g-3")

    return dbc.Container(fluid=True, style={"padding": "18px"}, children=[
        profile_filter_card,
        html.Div(style={"height": "14px"}),
        top_profile_row,
        html.Div(style={"height": "14px"}),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Win/Loss by Year", style={"fontWeight": "600", "marginBottom": "8px"}),
                dcc.Graph(id="p2_by_year", config={"displayModeBar": False}, style={"height": "320px"}),
            ])), width=6),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Win/Loss by Surface", style={"fontWeight": "600", "marginBottom": "8px"}),
                dcc.Graph(id="p2_by_surface", config={"displayModeBar": False}, style={"height": "320px"}),
            ])), width=6),
        ], className="g-3"),

        html.Div(style={"height": "14px"}),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Win/Loss by Tournament Type", style={"fontWeight": "600", "marginBottom": "8px"}),
                dcc.Graph(id="p2_by_type", config={"displayModeBar": False}, style={"height": "320px"}),
            ])), width=6),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Win/Loss by Round", style={"fontWeight": "600", "marginBottom": "8px"}),
                dcc.Graph(id="p2_by_round", config={"displayModeBar": False}, style={"height": "320px"}),
            ])), width=6),
        ], className="g-3"),

        html.Div(style={"height": "14px"}),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Top Opponents", style={"fontWeight": "600", "marginBottom": "8px"}),
                dash_table.DataTable(
                    id="p2_top_opps",
                    page_size=10, style_table={"overflowX": "auto"},
                    style_cell={"fontFamily": "Arial", "fontSize": "12px", "padding": "6px"},
                    style_header={"fontWeight": "600"},
                    sort_action="native",
                )
            ])), width=6),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Recent Matches", style={"fontWeight": "600", "marginBottom": "8px"}),
                dash_table.DataTable(
                    id="p2_recent",
                    page_size=10, style_table={"overflowX": "auto"},
                    style_cell={"fontFamily": "Arial", "fontSize": "12px", "padding": "6px"},
                    style_header={"fontWeight": "600"},
                    sort_action="native",
                )
            ])), width=6),
        ], className="g-3"),
    ])
