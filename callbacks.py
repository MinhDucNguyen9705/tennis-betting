# callbacks.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output

from data_access import (
    get_opponents_for_id, kpis, surface_wl,
    profile_kpis, total_aces_by_player, avg_attributes_by_player, wl_grouped,
    top_opponents, recent_matches_player,
    win_round_counts, COLS
)
from figures import stacked_surface_fig, stacked_wl_fig, double_donut_win_round, radar_figure
from data_access import detail_rows

def resolve_profile_filters(mode, start_date, end_date, level):
    if mode == "all":
        return None, None, "(All)"
    return start_date, end_date, level

def register_callbacks(app):

    @app.callback(
        Output("player2", "options"),
        Output("player2", "value"),
        Input("player1", "value"),
    )
    def update_player2_options(p1_id):
        if not p1_id:
            return [{"label": "(All)", "value": "(All)"}], "(All)"
        opps = get_opponents_for_id(p1_id)
        options = [{"label": "(All)", "value": "(All)"}] + [
            {"label": f"{r.opp_name} (ID: {r.opp_id})", "value": r.opp_id}
            for r in opps.itertuples(index=False)
            if pd.notna(r.opp_id)
        ]
        return options, "(All)"

    @app.callback(
        Output("kpi_matches", "children"),
        Output("kpi_winrate", "children"),
        Output("kpi_predrate", "children"),
        Output("surface_bar", "figure"),
        Output("skill_radar", "figure"),
        Output("detail_table", "data"),
        Output("detail_table", "columns"),
        Input("player1", "value"),
        Input("player2", "value"),
        Input("date_range", "start_date"),
        Input("date_range", "end_date"),
        Input("tour_level", "value"),
    )
    def refresh_h2h(p1_id, p2_id, start_date, end_date, level):
        if not p1_id:
            empty = go.Figure()
            return "0", "—", "—", empty, empty, [], []

        n, wr, pr = kpis(p1_id, p2_id, start_date, end_date, level)
        k1 = f"{n:,}"
        k2 = "—" if wr is None or np.isnan(wr) else f"{wr*100:.1f}%"
        k3 = "—" if pr is None or np.isnan(pr) else f"{pr*100:.1f}%"

        fig_bar = stacked_surface_fig(surface_wl(p1_id, p2_id, start_date, end_date, level))
        fig_radar = radar_figure(p1_id, p2_id, start_date, end_date, level)

        detail = detail_rows(p1_id, p2_id, start_date, end_date, level, limit=10000)
        cols = [{"name": c, "id": c} for c in detail.columns]
        data = detail.to_dict("records")
        return k1, k2, k3, fig_bar, fig_radar, data, cols

    @app.callback(
        Output("p2_kpi_matches", "children"),
        Output("p2_kpi_wr", "children"),
        Output("p2_kpi_form10", "children"),
        Output("p2_kpi_aces", "children"),
        Output("p2_kpi_svpt", "children"),
        Output("p2_kpi_bpSaved", "children"),
        Output("p2_win_round_donut", "figure"),
        Output("p2_by_year", "figure"),
        Output("p2_by_surface", "figure"),
        Output("p2_by_type", "figure"),
        Output("p2_by_round", "figure"),
        Output("p2_top_opps", "data"),
        Output("p2_top_opps", "columns"),
        Output("p2_recent", "data"),
        Output("p2_recent", "columns"),
        Input("p2_player", "value"),
        Input("p2_mode", "value"),
        Input("p2_date_range", "start_date"),
        Input("p2_date_range", "end_date"),
        Input("p2_tour_level", "value"),
    )
    def refresh_profile_tab(p_id, mode, start_date, end_date, level):
        if not p_id:
            empty = go.Figure()
            return (
                "0", "—", "—", "—", "—", "—",
                empty,                  
                empty, empty, empty, empty,    
                [], [],                        
                [], []                    
            )

        start_date, end_date, level = resolve_profile_filters(mode, start_date, end_date, level)

        n, wr, form10 = profile_kpis(p_id, start_date, end_date, level)
        k_matches = f"{n:,}"
        k_wr = "—" if wr is None or np.isnan(wr) else f"{wr*100:.1f}%"
        k_form = "—" if form10 is None or np.isnan(form10) else f"{form10*100:.1f}%"

        avg_aces, avg_svpt, avg_bpSaved = avg_attributes_by_player(p_id, start_date, end_date, level)
        k_aces = f"{avg_aces:.2f}" if avg_aces is not None else "—"
        k_svpt = f"{avg_svpt:.2f}" if avg_svpt is not None else "—"
        k_bpSaved = f"{avg_bpSaved:.2f}" if avg_bpSaved is not None else "—"

        fig_donut = double_donut_win_round(win_round_counts(p_id, start_date, end_date, level))

        fig_year = stacked_wl_fig(wl_grouped(p_id, start_date, end_date, level, "year"), "year", "Win/Loss by Year")
        fig_surf = stacked_wl_fig(wl_grouped(p_id, start_date, end_date, level, "tournament_surface"),
                                  "tournament_surface", "Win/Loss by Surface")

        # NOTE: your current code uses tournament_level for type; keep or change to tournament_type when you have it.
        df_type = wl_grouped(p_id, start_date, end_date, level, "tournament_level") if "tournament_level" in COLS else pd.DataFrame()
        fig_type = stacked_wl_fig(df_type, "tournament_level", "Win/Loss by Tournament Level") if not df_type.empty else go.Figure()

        df_round = wl_grouped(p_id, start_date, end_date, level, "round") if "round" in COLS else pd.DataFrame()
        fig_round = stacked_wl_fig(df_round, "round", "Win/Loss by Round") if not df_round.empty else go.Figure()

        opps = top_opponents(p_id, start_date, end_date, level, limit=50)
        opp_cols = [{"name": c, "id": c} for c in opps.columns]
        opp_data = opps.to_dict("records")

        rec = recent_matches_player(p_id, start_date, end_date, level, limit=30)
        rec_cols = [{"name": c, "id": c} for c in rec.columns]
        rec_data = rec.to_dict("records")

        return (k_matches, k_wr, k_form, k_aces, k_svpt, k_bpSaved,
                fig_donut, fig_year, fig_surf, fig_type, fig_round,
                opp_data, opp_cols, rec_data, rec_cols)

    @app.callback(
        Output("p2_date_range", "disabled"),
        Output("p2_tour_level", "disabled"),
        Input("p2_mode", "value")
    )
    def toggle_profile_filters(mode):
        disabled = (mode == "all")
        return disabled, disabled

    @app.callback(Output("p2_player_hint", "children"), Input("p2_player", "value"))
    def show_p2_player_id(pid):
        return f"ID: {pid}" if pid else ""
