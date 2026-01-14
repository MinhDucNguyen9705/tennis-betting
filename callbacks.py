# callbacks.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output

from data_access import (
    get_opponents_for_id, get_player_dim, kpis, surface_wl,
    profile_kpis, total_aces_by_player, avg_attributes_by_player, wl_grouped,
    top_opponents, recent_matches_player,
    win_round_counts, COLS
)

# Cache all players for dropdown (avoids repeated DB queries)
_all_players_cache = None
from figures import stacked_surface_fig, stacked_wl_fig, double_donut_win_round, radar_figure
from data_access import detail_rows
from h2h_inference import predict_h2h

def resolve_profile_filters(mode, start_date, end_date, level):
    if mode == "all":
        return None, None, "(All)"
    return start_date, end_date, level

def register_callbacks(app, player_dim=None):
    # Store player_dim in module scope for callback access
    _player_dim = player_dim

    @app.callback(
        Output("player2", "options"),
        Output("player2", "value"),
        Input("player1", "value"),
    )
    def update_player2_options(p1_id):
        if not p1_id:
            return [{"label": "(All)", "value": "(All)"}], "(All)"
        
        # Use pre-loaded player_dim passed from app.py
        if _player_dim is not None and not _player_dim.empty:
            options = [{"label": "(All)", "value": "(All)"}] + [
                {"label": f"{r.display_name} (ID: {r.id})", "value": r.id}
                for r in _player_dim.itertuples(index=False)
                if pd.notna(r.id) and r.id != p1_id
            ]
            print(f"[DEBUG] Player2 dropdown options: {len(options)} from pre-loaded player_dim")
        else:
            # Fallback to opponents only
            opps = get_opponents_for_id(p1_id)
            options = [{"label": "(All)", "value": "(All)"}] + [
                {"label": f"{r.opp_name} (ID: {r.opp_id})", "value": r.opp_id}
                for r in opps.itertuples(index=False)
                if pd.notna(r.opp_id)
            ]
            print(f"[DEBUG] Player2 dropdown options: {len(options)} fallback to opponents")
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
        Input("pred_model", "value"),
        Input("pred_surface", "value"),
        Input("pred_level", "value"),
        Input("pred_round", "value"),
        Input("pred_best_of", "value"),
    )
    def refresh_h2h(p1_id, p2_id, start_date, end_date, level,
                    pred_model, pred_surface, pred_level, pred_round, pred_best_of):
        if not p1_id:
            empty = go.Figure()
            return "0", "—", "—", empty, empty, [], []

        # Get match counts and historical win rate from database
        n, wr, _ = kpis(p1_id, p2_id, start_date, end_date, level)
        k1 = f"{n:,}"
        k2 = "—" if wr is None or np.isnan(wr) else f"{wr*100:.1f}%"
        
        # Get ML prediction (only when specific P2 is selected)
        k3 = "—"
        print(f"[DEBUG] Callback received: model={pred_model}, surface={pred_surface}, level={pred_level}, round={pred_round}, best_of={pred_best_of}")
        if p2_id and p2_id != "(All)" and pred_model and pred_model != "none":
            try:
                result = predict_h2h(
                    model_type=pred_model,
                    p1_id=p1_id,
                    p2_id=p2_id,
                    surface=pred_surface or "Hard",
                    level=pred_level or "M",
                    round_val=pred_round or "R32",
                    best_of=pred_best_of or 3
                )
                if result is not None:
                    # Handle both dict (new) and float (legacy) return types
                    if isinstance(result, dict):
                        prob = result.get('probability')
                        method = result.get('method', '')
                        if prob is not None:
                            k3 = f"{prob*100:.1f}% ({method})"
                    else:
                        # Legacy float return
                        k3 = f"{result*100:.1f}%"
            except Exception as e:
                print(f"ML prediction error: {e}")
                k3 = "—"

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
