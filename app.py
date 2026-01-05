# import numpy as np
# import pandas as pd
# import duckdb
# import math

# import dash
# from dash import dcc, html, dash_table, Input, Output
# import dash_bootstrap_components as dbc
# import plotly.graph_objects as go

# from data_access import (
#     get_player_dim,
#     get_levels,
#     get_date_bounds,
#     get_opponents_for_id,
#     kpis,
#     surface_wl,
#     detail_rows,
#     profile_kpis,
#     total_aces_by_player,
#     wl_grouped,
#     top_opponents,
#     recent_matches_player,
#     win_round_counts
# )

# from figures import radar_figure, stacked_surface_fig, stacked_wl_fig, double_donut_win_round
# from config import PARQUET_GLOB, LATEST_SKILLS, MEAN_SKILLS, FORM_SKILL, SKILLS, REQUIRED_BASE
# from db import con, existing_cols, sql_df

# # =========================
# # Build dropdown data
# # =========================
# player_dim = get_player_dim()
# levels = get_levels()
# min_dt, max_dt = get_date_bounds()

# default_p1 = player_dim["id"].iloc[0] if not player_dim.empty else None


# # =========================
# # Dash Layout
# # =========================
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# app.title = "Tennis Dashboard (Parquet + DuckDB)"


# def kpi_card(title, value_id):
#     return dbc.Card(
#         dbc.CardBody([
#             html.Div(title, style={"fontSize": "14px", "opacity": 0.8}),
#             html.Div(id=value_id, style={"fontSize": "28px", "fontWeight": "600", "marginTop": "6px"}),
#         ])
#     )


# dashboard_layout = dbc.Container(fluid=True, style={"padding": "18px"}, children=[

#     dbc.Row([
#         dbc.Col(
#             dbc.Card(
#                 dbc.CardBody([
#                     dbc.Row([
#                         dbc.Col(kpi_card("No. Match", "kpi_matches"), width=4),
#                         dbc.Col(kpi_card("Victory rate", "kpi_winrate"), width=4),
#                         dbc.Col(kpi_card("Pred Victory rate", "kpi_predrate"), width=4),
#                     ], className="g-2"),
#                 ]),
#             ),
#             width=8
#         ),

#         dbc.Col(
#             dbc.Card(
#                 dbc.CardBody([
#                     html.Div("Filter", style={"fontSize": "18px", "fontWeight": "600", "marginBottom": "10px"}),

#                     dbc.Row([
#                         dbc.Col([
#                             html.Div("Player 1", style={"fontSize": "12px", "opacity": 0.8}),
#                             dcc.Dropdown(
#                                 id="player1",
#                                 options=[
#                                     {"label": f"{r.display_name} (ID: {r.id})", "value": r.id}
#                                     for r in player_dim.itertuples(index=False)
#                                 ],
#                                 value=default_p1,
#                                 clearable=False
#                             )
#                         ], width=6, style={"minWidth": 0}),

#                         dbc.Col([
#                             html.Div("Player 2", style={"fontSize": "12px", "opacity": 0.8}),
#                             dcc.Dropdown(
#                                 id="player2",
#                                 options=[{"label": "(All)", "value": "(All)"}],
#                                 value="(All)",
#                                 clearable=False
#                             )
#                         ], width=6, style={"minWidth": 0}),
#                     ], className="g-2", style={"marginBottom": "10px"}),

#                     dbc.Row([
#                         dbc.Col([
#                             html.Div("Time Range", style={"fontSize": "12px", "opacity": 0.8}),
#                             html.Div(
#                                 dcc.DatePickerRange(
#                                     id="date_range",
#                                     min_date_allowed=min_dt,
#                                     max_date_allowed=max_dt,
#                                     start_date=min_dt,
#                                     end_date=max_dt,
#                                     display_format="YYYY-MM-DD",
#                                     number_of_months_shown=1,
#                                 ),
#                                 style={"width": "100%"}
#                             )
#                         ], width=8, style={"minWidth": 0}),

#                         dbc.Col([
#                             html.Div("Tour. Level", style={"fontSize": "12px", "opacity": 0.8}),
#                             dcc.Dropdown(
#                                 id="tour_level",
#                                 options=[{"label": "(All)", "value": "(All)"}] + [{"label": lv, "value": lv} for lv in levels],
#                                 value="(All)",
#                                 clearable=False
#                             )
#                         ], width=4, style={"minWidth": 0}),
#                     ], className="g-2"),
#                 ]),
#                 # IMPORTANT: no fixed height -> avoids overflow for date picker
#                 style={"height": "auto"}
#             ),
#             width=4
#         ),
#     ], className="g-3"),

#     html.Div(style={"height": "14px"}),

#     dbc.Row([
#         dbc.Col(
#             dbc.Card(
#                 dbc.CardBody([
#                     html.Div("Win/Loss by Surface (stacked)", style={"fontWeight": "600", "marginBottom": "8px"}),
#                     dcc.Graph(id="surface_bar", config={"displayModeBar": False}, style={"height": "340px"})
#                 ]),
#                 style={"height": "420px"}
#             ),
#             width=8
#         ),
#         dbc.Col(
#             dbc.Card(
#                 dbc.CardBody([
#                     html.Div("Skill Radar (Player 1 vs Player 2)", style={"fontWeight": "600", "marginBottom": "8px"}),
#                     dcc.Graph(id="skill_radar", config={"displayModeBar": False}, style={"height": "340px"})
#                 ]),
#                 style={"height": "420px"}
#             ),
#             width=4
#         ),
#     ], className="g-3"),

#     html.Div(style={"height": "14px"}),

#     dbc.Row([
#         dbc.Col(
#             dbc.Card(
#                 dbc.CardBody([
#                     html.Div("Detail table", style={"fontWeight": "600", "marginBottom": "8px"}),
#                     dash_table.DataTable(
#                         id="detail_table",
#                         page_size=12,
#                         style_table={"overflowX": "auto"},
#                         style_cell={"fontFamily": "Arial", "fontSize": "12px", "padding": "6px"},
#                         style_header={"fontWeight": "600"},
#                         sort_action="native",
#                         filter_action="native",
#                     )
#                 ]),
#                 style={"height": "420px"}
#             ),
#             width=12
#         )
#     ])
# ])

# profile_filter_card = dbc.Card(
#     dbc.CardBody([
#         html.Div("Player Profile Filters", style={"fontSize": "18px", "fontWeight": "600", "marginBottom": "10px"}),

#         dbc.Row([
#             dbc.Col([
#                 html.Div("Player", style={"fontSize": "12px", "opacity": 0.8}),
#                 dcc.Dropdown(
#                     id="p2_player",
#                     options=[{"label": r.display_name, "value": r.id}
#                              for r in player_dim.itertuples(index=False)],
#                     value=default_p1,
#                     clearable=False,
#                     searchable=True,
#                     placeholder="Select player..."
#                 ),
#                 html.Div(id="p2_player_hint", style={"fontSize": "12px", "opacity": 0.7, "marginTop": "4px"})
#             ], width=6),

#             dbc.Col([
#                 html.Div("Mode", style={"fontSize": "12px", "opacity": 0.8}),
#                 dbc.RadioItems(
#                     id="p2_mode",
#                     options=[
#                         {"label": "All time", "value": "all"},
#                         {"label": "Use filters below", "value": "filter"},
#                     ],
#                     value="all",
#                     inline=True
#                 ),
#             ], width=6),
#         ], className="g-2", style={"marginBottom": "10px"}),

#         dbc.Row([
#             dbc.Col([
#                 html.Div("Time Range", style={"fontSize": "12px", "opacity": 0.8}),
#                 html.Div(
#                     dcc.DatePickerRange(
#                         id="p2_date_range",
#                         min_date_allowed=min_dt,
#                         max_date_allowed=max_dt,
#                         start_date=min_dt,
#                         end_date=max_dt,
#                         display_format="YYYY-MM-DD",
#                         number_of_months_shown=1,
#                     ),
#                     style={"width": "100%"}
#                 )
#             ], width=8),

#             dbc.Col([
#                 html.Div("Tour. Level", style={"fontSize": "12px", "opacity": 0.8}),
#                 dcc.Dropdown(
#                     id="p2_tour_level",
#                     options=[{"label": "(All)", "value": "(All)"}] + [{"label": lv, "value": lv} for lv in levels],
#                     value="(All)",
#                     clearable=False
#                 )
#             ], width=4),
#         ], className="g-2"),
#     ])
# )

# top_profile_row = dbc.Row(
#     [
#         # LEFT: Matches + Win rate (stacked)
#         dbc.Col(
#             dbc.Row(
#                 [
#                     dbc.Col(dbc.Card(dbc.CardBody([
#                         html.Div("Matches (All time)", style={"fontSize": "14px", "opacity": 0.8}),
#                         html.Div(id="p2_kpi_matches", style={"fontSize": "28px", "fontWeight": "600", "marginTop": "6px", "height": "80px"}),
#                     ])), width=12),

#                     dbc.Col(dbc.Card(dbc.CardBody([
#                         html.Div("Win rate", style={"fontSize": "14px", "opacity": 0.8}),
#                         html.Div(id="p2_kpi_wr", style={"fontSize": "28px", "fontWeight": "600", "marginTop": "6px", "height": "80px"}),
#                     ])), width=12),
#                 ],
#                 className="g-3"
#             ),
#             width=4
#         ),

#         # MIDDLE: Donut chart
#         dbc.Col(
#             dbc.Card(
#                 dbc.CardBody([
#                     html.Div("Win/Loss + Round Breakdown", style={"fontWeight": "600", "marginBottom": "8px"}),
#                     dcc.Graph(
#                         id="p2_win_round_donut",
#                         config={"displayModeBar": False},
#                         style={"height": "240px"}  # adjust if you want taller
#                     ),
#                 ])
#             ),
#             width=4
#         ),

#         # RIGHT: Form + Total aces (stacked)
#         dbc.Col(
#             dbc.Row(
#                 [
#                     dbc.Col(dbc.Card(dbc.CardBody([
#                         html.Div("Form win rate (last 10)", style={"fontSize": "14px", "opacity": 0.8}),
#                         html.Div(id="p2_kpi_form10", style={"fontSize": "28px", "fontWeight": "600", "marginTop": "6px", "height": "80px"}),
#                     ])), width=12),

#                     dbc.Col(dbc.Card(dbc.CardBody([
#                         html.Div("Total aces", style={"fontSize": "14px", "opacity": 0.8}),
#                         html.Div(id="p2_kpi_aces", style={"fontSize": "28px", "fontWeight": "600", "marginTop": "6px", "height": "80px"}),
#                     ])), width=12),
#                 ],
#                 className="g-3"
#             ),
#             width=4
#         ),
#     ],
#     className="g-3"
# )

# player_profile_layout = dbc.Container(
#     fluid=True,
#     style={"padding": "18px"},
#     children=[
#         profile_filter_card,
#         html.Div(style={"height": "14px"}),
#         top_profile_row,
#         html.Div(style={"height": "14px"}),

#         dbc.Row([
#             dbc.Col(dbc.Card(dbc.CardBody([
#                 html.Div("Win/Loss by Year", style={"fontWeight": "600", "marginBottom": "8px"}),
#                 dcc.Graph(id="p2_by_year", config={"displayModeBar": False}, style={"height": "320px"}),
#             ])), width=6),

#             dbc.Col(dbc.Card(dbc.CardBody([
#                 html.Div("Win/Loss by Surface", style={"fontWeight": "600", "marginBottom": "8px"}),
#                 dcc.Graph(id="p2_by_surface", config={"displayModeBar": False}, style={"height": "320px"}),
#             ])), width=6),
#         ], className="g-3"),

#         html.Div(style={"height": "14px"}),

#         dbc.Row([
#             dbc.Col(dbc.Card(dbc.CardBody([
#                 html.Div("Win/Loss by Tournament Type", style={"fontWeight": "600", "marginBottom": "8px"}),
#                 dcc.Graph(id="p2_by_type", config={"displayModeBar": False}, style={"height": "320px"}),
#             ])), width=6),

#             dbc.Col(dbc.Card(dbc.CardBody([
#                 html.Div("Win/Loss by Round", style={"fontWeight": "600", "marginBottom": "8px"}),
#                 dcc.Graph(id="p2_by_round", config={"displayModeBar": False}, style={"height": "320px"}),
#             ])), width=6),
#         ], className="g-3"),

#         html.Div(style={"height": "14px"}),

#         dbc.Row([
#             dbc.Col(dbc.Card(dbc.CardBody([
#                 html.Div("Top Opponents", style={"fontWeight": "600", "marginBottom": "8px"}),
#                 dash_table.DataTable(
#                     id="p2_top_opps",
#                     page_size=10,
#                     style_table={"overflowX": "auto"},
#                     style_cell={"fontFamily": "Arial", "fontSize": "12px", "padding": "6px"},
#                     style_header={"fontWeight": "600"},
#                     sort_action="native",
#                 )
#             ])), width=6),

#             dbc.Col(dbc.Card(dbc.CardBody([
#                 html.Div("Recent Matches", style={"fontWeight": "600", "marginBottom": "8px"}),
#                 dash_table.DataTable(
#                     id="p2_recent",
#                     page_size=10,
#                     style_table={"overflowX": "auto"},
#                     style_cell={"fontFamily": "Arial", "fontSize": "12px", "padding": "6px"},
#                     style_header={"fontWeight": "600"},
#                     sort_action="native",
#                 )
#             ])), width=6),
#         ], className="g-3"),
#     ]
# )

# app.layout = dbc.Container(
#     fluid=True,
#     children=[
#         dbc.Tabs(
#             [
#                 dbc.Tab(dashboard_layout, label="Head to Head", tab_id="tab-h2h"),
#                 dbc.Tab(player_profile_layout, label="Player Profile (vs All)", tab_id="tab-profile"),
#             ],
#             active_tab="tab-h2h",
#         )
#     ]
# )

# # =========================
# # Callbacks
# # =========================
# @app.callback(
#     Output("player2", "options"),
#     Output("player2", "value"),
#     Input("player1", "value"),
# )
# def update_player2_options(p1_id):
#     if not p1_id:
#         return [{"label": "(All)", "value": "(All)"}], "(All)"
#     opps = get_opponents_for_id(p1_id)
#     options = [{"label": "(All)", "value": "(All)"}] + [
#         {"label": f"{r.opp_name} (ID: {r.opp_id})", "value": r.opp_id}
#         for r in opps.itertuples(index=False)
#         if pd.notna(r.opp_id)
#     ]
#     return options, "(All)"


# @app.callback(
#     Output("kpi_matches", "children"),
#     Output("kpi_winrate", "children"),
#     Output("kpi_predrate", "children"),
#     Output("surface_bar", "figure"),
#     Output("skill_radar", "figure"),
#     Output("detail_table", "data"),
#     Output("detail_table", "columns"),
#     Input("player1", "value"),
#     Input("player2", "value"),
#     Input("date_range", "start_date"),
#     Input("date_range", "end_date"),
#     Input("tour_level", "value"),
# )
# def refresh(p1_id, p2_id, start_date, end_date, level):
#     if not p1_id:
#         empty = go.Figure()
#         return "0", "—", "—", empty, empty, [], []

#     n, wr, pr = kpis(p1_id, p2_id, start_date, end_date, level)
#     k1 = f"{n:,}"
#     k2 = "—" if wr is None or np.isnan(wr) else f"{wr*100:.1f}%"
#     k3 = "—" if pr is None or np.isnan(pr) else f"{pr*100:.1f}%"

#     surf = surface_wl(p1_id, p2_id, start_date, end_date, level)
#     fig_bar = stacked_surface_fig(surf)

#     fig_radar = radar_figure(p1_id, p2_id, start_date, end_date, level)

#     detail = detail_rows(p1_id, p2_id, start_date, end_date, level, limit=10000)
#     cols = [{"name": c, "id": c} for c in detail.columns]
#     data = detail.to_dict("records")

#     return k1, k2, k3, fig_bar, fig_radar, data, cols

# def resolve_profile_filters(mode, start_date, end_date, level):
#     """
#     mode='all'    -> ignore date + level filters
#     mode='filter' -> use date + level from Tab 2 controls
#     """
#     if mode == "all":
#         return None, None, "(All)"
#     return start_date, end_date, level

# @app.callback(
#     Output("p2_kpi_matches", "children"),
#     Output("p2_kpi_wr", "children"),
#     Output("p2_kpi_form10", "children"),
#     Output("p2_kpi_aces", "children"),
#     Output("p2_win_round_donut", "figure"),

#     Output("p2_by_year", "figure"),
#     Output("p2_by_surface", "figure"),
#     Output("p2_by_type", "figure"),
#     Output("p2_by_round", "figure"),

#     Output("p2_top_opps", "data"),
#     Output("p2_top_opps", "columns"),
#     Output("p2_recent", "data"),
#     Output("p2_recent", "columns"),

#     Input("p2_player", "value"),
#     Input("p2_mode", "value"),
#     Input("p2_date_range", "start_date"),
#     Input("p2_date_range", "end_date"),
#     Input("p2_tour_level", "value"),
# )
# def refresh_profile(p_id, mode, start_date, end_date, level):
#     if not p_id:
#         empty = go.Figure()
#         return (
#             "0", "—", "—", "—",
#             empty, empty, empty, empty, empty,
#             [], [], [], []
#         )

#     # Apply "All time" vs "Use filters"
#     start_date, end_date, level = resolve_profile_filters(mode, start_date, end_date, level)

#     n, wr, form10 = profile_kpis(p_id, start_date, end_date, level)

#     k_matches = f"{n:,}"
#     k_wr = "—" if wr is None or np.isnan(wr) else f"{wr*100:.1f}%"
#     k_form = "—" if form10 is None or np.isnan(form10) else f"{form10*100:.1f}%"

#     aces_sum = total_aces_by_player(p_id, start_date, end_date, level)
#     k_cov = f"{aces_sum:,}" if aces_sum is not None else "0"

#     df_year  = wl_grouped(p_id, start_date, end_date, level, "year")
#     df_surf  = wl_grouped(p_id, start_date, end_date, level, "tournament_surface")
#     df_type  = wl_grouped(p_id, start_date, end_date, level, "tournament_level") if "tournament_level" in COLS else pd.DataFrame()
#     df_round = wl_grouped(p_id, start_date, end_date, level, "round") if "round" in COLS else pd.DataFrame()

#     fig_year = stacked_wl_fig(df_year, "year", "Win/Loss by Year")
#     fig_surf = stacked_wl_fig(df_surf, "tournament_surface", "Win/Loss by Surface")
#     fig_type = stacked_wl_fig(df_type, "tournament_level", "Win/Loss by Tournament Level") if not df_type.empty else go.Figure()
#     fig_round = stacked_wl_fig(df_round, "round", "Win/Loss by Round") if not df_round.empty else go.Figure()

#     opps = top_opponents(p_id, start_date, end_date, level, limit=50)
#     opp_cols = [{"name": c, "id": c} for c in opps.columns]
#     opp_data = opps.to_dict("records")

#     rec = recent_matches_player(p_id, start_date, end_date, level, limit=30)
#     rec_cols = [{"name": c, "id": c} for c in rec.columns]
#     rec_data = rec.to_dict("records")

#     df_wr = win_round_counts(p_id, start_date, end_date, level)
#     fig_donut = double_donut_win_round(df_wr)

#     return (
#         k_matches, k_wr, k_form, k_cov,
#         fig_donut, fig_year, fig_surf, fig_type, fig_round,
#         opp_data, opp_cols,
#         rec_data, rec_cols
#     )

# @app.callback(
#     Output("p2_date_range", "disabled"),
#     Output("p2_tour_level", "disabled"),
#     Input("p2_mode", "value")
# )
# def toggle_profile_filters(mode):
#     disabled = (mode == "all")
#     return disabled, disabled

# if __name__ == "__main__":
#     app.run(debug=True, host="127.0.0.1", port=8050)

import dash
import dash_bootstrap_components as dbc

from data_access import get_player_dim, get_levels, get_date_bounds
from layouts import make_h2h_layout, make_profile_layout
from callbacks import register_callbacks
from backtest.backtest_tab import make_backtest_layout, register_backtest_callbacks, read_output_model

player_dim = get_player_dim()
levels = get_levels()
min_dt, max_dt = get_date_bounds()
default_p1 = player_dim["id"].iloc[0] if not player_dim.empty else None

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Tennis Dashboard (Parquet + DuckDB)"

dashboard_layout = make_h2h_layout(player_dim, levels, min_dt, max_dt, default_p1)
profile_layout = make_profile_layout(player_dim, levels, min_dt, max_dt, default_p1)

app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Tabs(
            [
                dbc.Tab(dashboard_layout, label="Head to Head", tab_id="tab-h2h"),
                dbc.Tab(profile_layout, label="Player Profile (vs All)", tab_id="tab-profile"),
                dbc.Tab(make_backtest_layout(), label="Backtest"),
            ],
            active_tab="tab-h2h",
        )
    ],
)

register_callbacks(app, player_dim)
def load_backtest_df():
    # IMPORTANT: replace this with your real data source:
    # - DuckDB query
    # - Parquet scan
    # - or your existing function that returns the matches dataframe used for modeling/backtest
    #
    # Example:
    # return sql_df("SELECT * FROM matches WHERE pred_prob IS NOT NULL AND odds IS NOT NULL")
    return read_output_model()  # placeholder

register_backtest_callbacks(app, load_backtest_df)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)