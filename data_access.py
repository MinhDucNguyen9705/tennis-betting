# data_access.py
import math
import numpy as np
import pandas as pd

from db import sql_df, existing_cols
from config import REQUIRED_BASE, SKILLS, MEAN_SKILLS, LATEST_SKILLS

COLS = existing_cols()

missing_base = REQUIRED_BASE - COLS
if missing_base:
    print("WARNING: Missing required columns:", missing_base)

def get_date_bounds():
    # tournament_date should be DATE/TIMESTAMP in parquet; cast safely
    q = """
    SELECT
      MIN(CAST(tournament_date AS DATE)) AS min_dt,
      MAX(CAST(tournament_date AS DATE)) AS max_dt
    FROM matches
    """
    r = sql_df(q).iloc[0]
    mn = pd.to_datetime(r["min_dt"], errors="coerce")
    mx = pd.to_datetime(r["max_dt"], errors="coerce")
    if pd.isna(mn):
        mn = pd.Timestamp("2000-01-01")
    if pd.isna(mx):
        mx = pd.Timestamp.today()
    return mn.date(), mx.date()


def get_levels() -> list[str]:
    if "tournament_level" not in COLS:
        return []
    q = """
    SELECT DISTINCT tournament_level
    FROM matches
    WHERE tournament_level IS NOT NULL
    ORDER BY tournament_level
    """
    return sql_df(q)["tournament_level"].tolist()


def get_player_dim() -> pd.DataFrame:
    # Use latest seen name for each ID for display (arg_max)
    q = """
    WITH u AS (
      SELECT ID_1 AS id, Name_1 AS name, CAST(tournament_date AS DATE) AS dt FROM matches
      UNION ALL
      SELECT ID_2 AS id, Name_2 AS name, CAST(tournament_date AS DATE) AS dt FROM matches
    )
    SELECT
      id,
      arg_max(name, dt) AS display_name
    FROM u
    WHERE id IS NOT NULL
    GROUP BY id
    ORDER BY display_name
    """
    return sql_df(q)


def get_opponents_for_id(pid) -> pd.DataFrame:
    q = """
    SELECT DISTINCT
      CASE WHEN ID_1 = $p THEN ID_2 ELSE ID_1 END AS opp_id,
      CASE WHEN ID_1 = $p THEN Name_2 ELSE Name_1 END AS opp_name
    FROM matches
    WHERE ID_1 = $p OR ID_2 = $p
    ORDER BY opp_name
    """
    return sql_df(q, {"p": pid})


def build_where(p1_id, p2_id, start_date, end_date, level):
    where = ["(ID_1 = $p1 OR ID_2 = $p1)"]
    params = {"p1": p1_id}

    if p2_id and p2_id != "(All)":
        where.append("((ID_1 = $p1 AND ID_2 = $p2) OR (ID_1 = $p2 AND ID_2 = $p1))")
        params["p2"] = p2_id

    # Handle partial edits safely
    if start_date and end_date:
        where.append("CAST(tournament_date AS DATE) BETWEEN $start AND $end")
        params["start"] = str(start_date)
        params["end"] = str(end_date)
    elif start_date and not end_date:
        where.append("CAST(tournament_date AS DATE) >= $start")
        params["start"] = str(start_date)
    elif end_date and not start_date:
        where.append("CAST(tournament_date AS DATE) <= $end")
        params["end"] = str(end_date)

    if level and level != "(All)" and "tournament_level" in COLS:
        where.append("tournament_level = $lvl")
        params["lvl"] = level

    return " AND ".join(where), params


# =========================
# GLOBAL SKILL RANGES (computed once)
# =========================
def compute_global_ranges_mixed(skills: dict, player_level_skills: set[str]) -> dict:
    ranges = {}

    for col in skills.keys():
        c1, c2 = f"{col}_1", f"{col}_2"
        if c1 not in COLS or c2 not in COLS:
            continue

        if col in player_level_skills:
            # player-level range
            q = f"""
            WITH player_match AS (
              SELECT ID_1 AS pid, {c1} AS v FROM matches WHERE ID_1 IS NOT NULL AND {c1} IS NOT NULL
              UNION ALL
              SELECT ID_2 AS pid, {c2} AS v FROM matches WHERE ID_2 IS NOT NULL AND {c2} IS NOT NULL
            ),
            per_player AS (
              SELECT pid, AVG(v) AS v_player
              FROM player_match
              GROUP BY pid
            )
            SELECT MIN(v_player) AS mn, MAX(v_player) AS mx, AVG(v_player) AS mean
            FROM per_player
            """
        else:
            # match-level range
            q = f"""
            SELECT MIN(v) AS mn, MAX(v) AS mx, AVG(v) AS mean
            FROM (
              SELECT {c1} AS v FROM matches
              UNION ALL
              SELECT {c2} AS v FROM matches
            )
            WHERE v IS NOT NULL
            """

        r = sql_df(q).iloc[0]
        ranges[col] = {"min": r["mn"], "max": r["mx"], "mean": r["mean"]}

    return ranges


GLOBAL_RANGES = compute_global_ranges_mixed(SKILLS, MEAN_SKILLS)

def kpis(p1_id, p2_id, start_date, end_date, level):
    where, params = build_where(p1_id, p2_id, start_date, end_date, level)

    # "Pred Victory rate" optional: if Ranking_Points exists, compute proxy; else return NULL
    has_rp = ("Ranking_Points_1" in COLS) and ("Ranking_Points_2" in COLS)

    pred_sql = "NULL AS pred_prob_proxy"
    if has_rp:
        pred_sql = """
        CASE
          WHEN Ranking_Points_1 IS NULL OR Ranking_Points_2 IS NULL THEN NULL
          WHEN ID_1 = $p1 THEN 1.0 / (1.0 + POW(10.0, (Ranking_Points_2 - Ranking_Points_1)/400.0))
          WHEN ID_2 = $p1 THEN 1.0 / (1.0 + POW(10.0, (Ranking_Points_1 - Ranking_Points_2)/400.0))
          ELSE NULL
        END AS pred_prob_proxy
        """

    q = f"""
    WITH filtered AS (
      SELECT
        Winner, ID_1, ID_2,
        {pred_sql}
      FROM matches
      WHERE {where}
    )
    SELECT
      COUNT(*) AS n_matches,
      AVG(
        CASE
          WHEN (ID_1 = $p1 AND Winner = 0) OR (ID_2 = $p1 AND Winner = 1) THEN 1.0
          ELSE 0.0
        END
      ) AS win_rate,
      AVG(pred_prob_proxy) AS pred_rate
    FROM filtered
    """
    r = sql_df(q, params).iloc[0]
    return int(r["n_matches"]), r["win_rate"], r["pred_rate"]


def surface_wl(p1_id, p2_id, start_date, end_date, level) -> pd.DataFrame:
    where, params = build_where(p1_id, p2_id, start_date, end_date, level)

    q = f"""
    WITH base AS (
      SELECT tournament_surface,
             CASE
               WHEN (ID_1 = $p1 AND Winner = 0) OR (ID_2 = $p1 AND Winner = 1) THEN 1
               ELSE 0
             END AS is_win
      FROM matches
      WHERE {where}
        AND tournament_surface IS NOT NULL
    )
    SELECT tournament_surface,
           SUM(is_win) AS wins,
           SUM(1 - is_win) AS losses,
           AVG(is_win*1.0) AS win_rate
    FROM base
    GROUP BY tournament_surface
    ORDER BY (wins + losses) DESC
    """
    return sql_df(q, params)


def latest_match_skills_by_id(pid, start_date, end_date, level):
    where = ["(ID_1 = $p OR ID_2 = $p)"]
    params = {"p": pid}

    if start_date and end_date:
        where.append("CAST(tournament_date AS DATE) BETWEEN $start AND $end")
        params["start"] = str(start_date)
        params["end"] = str(end_date)

    if level and level != "(All)":
        where.append("tournament_level = $lvl")
        params["lvl"] = level

    # Only select the needed columns (faster)
    select_cols = ["ID_1", "ID_2", "Name_1", "Name_2", "Winner", "tournament_date"]
    for k in LATEST_SKILLS:
        c1, c2 = f"{k}_1", f"{k}_2"
        if c1 in COLS and c2 in COLS:
            select_cols += [c1, c2]

    q = f"""
    SELECT {", ".join(select_cols)}
    FROM matches
    WHERE {" AND ".join(where)}
    ORDER BY CAST(tournament_date AS DATE) DESC
    LIMIT 1
    """
    df = sql_df(q, params)
    if df.empty:
        return None

    row = df.iloc[0]
    if row["ID_1"] == pid:
        suf = "_1"
        display_name = row["Name_1"]
    else:
        suf = "_2"
        display_name = row["Name_2"]

    out = {"display_name": display_name}

    # Get latest-match stats from the correct side
    for k in LATEST_SKILLS:
        c = f"{k}{suf}"
        if c in df.columns:
            out[k] = row[c]
        else:
            out[k] = None

    # If you want Win Rate as 0/1 from match outcome (recommended)
    out["Victories_Percentage"] = 1.0 if (
        (row["ID_1"] == pid and row["Winner"] == 0) or (row["ID_2"] == pid and row["Winner"] == 1)
    ) else 0.0

    return out

def mean_skills_by_id(pid, start_date, end_date, level):
    where = ["(ID_1 = $p OR ID_2 = $p)"]
    params = {"p": pid}

    if start_date and end_date:
        where.append("CAST(tournament_date AS DATE) BETWEEN $start AND $end")
        params["start"] = str(start_date)
        params["end"] = str(end_date)

    if level and level != "(All)":
        where.append("tournament_level = $lvl")
        params["lvl"] = level

    agg_exprs = []
    for k in MEAN_SKILLS:
        c1, c2 = f"{k}_1", f"{k}_2"
        if c1 in COLS and c2 in COLS:
            agg_exprs.append(f"AVG(CASE WHEN ID_1=$p THEN {c1} ELSE {c2} END) AS {k}")

    if not agg_exprs:
        return {}

    q = f"""
    SELECT {", ".join(agg_exprs)}
    FROM matches
    WHERE {" AND ".join(where)}
    """
    row = sql_df(q, params).iloc[0].to_dict()
    return row

def last_n_win_rate_by_id(pid, n=10, start_date=None, end_date=None, level="(All)"):
    where = ["(ID_1 = $p OR ID_2 = $p)"]
    params = {"p": pid}

    if start_date and end_date:
        where.append("CAST(tournament_date AS DATE) BETWEEN $start AND $end")
        params["start"] = str(start_date)
        params["end"] = str(end_date)

    if level and level != "(All)":
        where.append("tournament_level = $lvl")
        params["lvl"] = level

    q = f"""
    WITH recent AS (
      SELECT
        CASE
          WHEN (ID_1 = $p AND Winner = 0)
            OR (ID_2 = $p AND Winner = 1)
          THEN 1.0
          ELSE 0.0
        END AS is_win
      FROM matches
      WHERE {" AND ".join(where)}
      ORDER BY CAST(tournament_date AS DATE) DESC
      LIMIT {int(n)}
    )
    SELECT
      COUNT(*) AS n_games,
      AVG(is_win) AS win_rate
    FROM recent
    """
    df = sql_df(q, params)
    if df.empty or df.iloc[0]["n_games"] == 0:
        return None
    return float(df.iloc[0]["win_rate"])

def normalize(val, rmin, rmax):
    if pd.isna(val) or rmin is None or rmax is None or rmax == rmin:
        return 0.5
    return float((val - rmin) / (rmax - rmin))

def radar_values(pid, start_date, end_date, level):
    latest = latest_match_skills_by_id(pid, start_date, end_date, level)
    if latest is None:
        return None, None, None

    means = mean_skills_by_id(pid, start_date, end_date, level)

    # ✅ correct form win rate
    win_rate_10 = last_n_win_rate_by_id(
        pid, n=10, start_date=start_date, end_date=end_date, level=level
    )

    labels = []
    values = []

    for k, label in SKILLS.items():
        labels.append(label)

        # ---- FORM: win rate (last 10 matches) ----
        if k == "Victories_Percentage":
            if win_rate_10 is None:
                norm = 0.5
            else:
                norm = float(np.clip(win_rate_10, 0.0, 1.0))

        # ---- LATEST MATCH SKILLS ----
        elif k in LATEST_SKILLS:
            raw = latest.get(k)
            r = GLOBAL_RANGES.get(k)
            if raw is None or pd.isna(raw) or r is None or r["max"] == r["min"]:
                norm = 0.5
            else:
                norm = (raw - r["min"]) / (r["max"] - r["min"])

        # ---- MEAN SKILLS (stamina) ----
        elif k in MEAN_SKILLS:
            raw = means.get(k)
            r = GLOBAL_RANGES.get(k)
            if raw is None or pd.isna(raw) or r is None or r["max"] == r["min"]:
                norm = 0.5
            else:
                norm = (raw - r["min"]) / (r["max"] - r["min"])

        else:
            norm = 0.5

        values.append(float(np.clip(norm, 0.0, 1.0)))

    return latest["display_name"], labels, values

def detail_rows(p1_id, p2_id, start_date, end_date, level, limit=2000) -> pd.DataFrame:
    where, params = build_where(p1_id, p2_id, start_date, end_date, level)

    # pick columns safely
    want = [
        "tournament_date", "tournament", "tournament_level", "tournament_surface", "round",
        "ID_1", "Name_1", "ID_2", "Name_2", "Winner", "score", "elapsed_minutes", 'aces_nb_1', 'doublefaults_nb_1', 'svpt_1',
        '1stIn_1', '1stWon_1', '2ndWon_1', 'SvGms_1', 'bpSaved_1', 'bpFaced_1',
        'aces_nb_2', 'doublefaults_nb_2', 'svpt_2', '1stIn_2', '1stWon_2',
        '2ndWon_2', 'SvGms_2', 'bpSaved_2', 'bpFaced_2'
    ]
    want = [c for c in want if c in COLS]

    q = f"""
    SELECT {", ".join(want)}
    FROM matches
    WHERE {where}
    ORDER BY CAST(tournament_date AS DATE) DESC
    LIMIT {int(limit)}
    """
    df = sql_df(q, params)
    if df.empty:
        return df

    df["Result_for_P1"] = np.where(
        ((df["ID_1"] == p1_id) & (df["Winner"] == 0)) | ((df["ID_2"] == p1_id) & (df["Winner"] == 1)),
        "Win", "Loss"
    )
    df["tournament_date"] = pd.to_datetime(df["tournament_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df

def build_where_player_only(p_id, start_date, end_date, level):
    where = ["(ID_1 = $p OR ID_2 = $p)"]
    params = {"p": p_id}

    if start_date and end_date:
        where.append("CAST(tournament_date AS DATE) BETWEEN $start AND $end")
        params["start"] = str(start_date)
        params["end"] = str(end_date)
    elif start_date and not end_date:
        where.append("CAST(tournament_date AS DATE) >= $start")
        params["start"] = str(start_date)
    elif end_date and not start_date:
        where.append("CAST(tournament_date AS DATE) <= $end")
        params["end"] = str(end_date)

    if level and level != "(All)" and "tournament_level" in COLS:
        where.append("tournament_level = $lvl")
        params["lvl"] = level

    return " AND ".join(where), params

def profile_kpis(p_id, start_date, end_date, level):
    where, params = build_where_player_only(p_id, start_date, end_date, level)

    has_type = "tournament_level" in COLS

    q = f"""
    WITH base AS (
      SELECT
        tournament_surface,
        {"tournament_level," if has_type else ""}
        CASE
          WHEN (ID_1 = $p AND Winner = 0) OR (ID_2 = $p AND Winner = 1) THEN 1.0
          ELSE 0.0
        END AS is_win
      FROM matches
      WHERE {where}
    ),
    recent AS (
      SELECT is_win
      FROM base
      ORDER BY is_win IS NULL, 1  -- dummy (we’ll re-order by date below)
    )
    SELECT
      (SELECT COUNT(*) FROM base) AS n_matches,
      (SELECT AVG(is_win) FROM base) AS win_rate,
      (SELECT COUNT(DISTINCT tournament_surface) FROM base) AS n_surfaces
      {", (SELECT COUNT(DISTINCT tournament_level) FROM base) AS n_levels" if has_type else ""}
    """
    # Better: compute form10 separately with date ordering:
    q2 = f"""
    WITH base AS (
      SELECT
        CAST(tournament_date AS DATE) AS dt,
        CASE
          WHEN (ID_1 = $p AND Winner = 0) OR (ID_2 = $p AND Winner = 1) THEN 1.0
          ELSE 0.0
        END AS is_win
      FROM matches
      WHERE {where}
      ORDER BY dt DESC
      LIMIT 10
    )
    SELECT AVG(is_win) AS form10, COUNT(*) AS n10 FROM base
    """
    r = sql_df(q, params).iloc[0]
    r2 = sql_df(q2, params).iloc[0]

    n_matches = int(r["n_matches"]) if r["n_matches"] is not None else 0
    win_rate = r["win_rate"]

    form10 = r2["form10"] if r2["n10"] and r2["n10"] > 0 else None
    return n_matches, win_rate, form10

def total_aces_by_player(pid, start_date=None, end_date=None, level="(All)"):
    # Requires columns: ID_1, ID_2, aces_nb_1, aces_nb_2
    if not all(c in COLS for c in ["ID_1", "ID_2", "aces_nb_1", "aces_nb_2"]):
        return None

    where = ["(ID_1 = $p OR ID_2 = $p)"]
    params = {"p": pid}

    if start_date and end_date:
        where.append("CAST(tournament_date AS DATE) BETWEEN $start AND $end")
        params["start"] = str(start_date)
        params["end"] = str(end_date)

    if level and level != "(All)" and "tournament_level" in COLS:
        where.append("tournament_level = $lvl")
        params["lvl"] = level

    q = f"""
    SELECT
      SUM(CASE WHEN ID_1 = $p THEN aces_nb_1 ELSE aces_nb_2 END) AS total_aces
    FROM matches
    WHERE {" AND ".join(where)}
    """
    df = sql_df(q, params)
    if df.empty:
        return 0
    v = df.iloc[0]["total_aces"]
    return int(v) if v is not None and not math.isnan(v) else 0

def avg_attributes_by_player(pid, start_date=None, end_date=None, level="(All)"):
    # Requires columns: ID_1, ID_2, aces_nb_1, aces_nb_2
    if not all(c in COLS for c in ["ID_1", "ID_2", "aces_nb_1", "aces_nb_2"]):
        return None

    where = ["(ID_1 = $p OR ID_2 = $p)"]
    params = {"p": pid}

    if start_date and end_date:
        where.append("CAST(tournament_date AS DATE) BETWEEN $start AND $end")
        params["start"] = str(start_date)
        params["end"] = str(end_date)

    if level and level != "(All)" and "tournament_level" in COLS:
        where.append("tournament_level = $lvl")
        params["lvl"] = level

    q = f"""
    SELECT
      AVG(CASE WHEN ID_1 = $p THEN aces_nb_1 ELSE aces_nb_2 END) AS avg_aces,
      AVG(CASE WHEN ID_1 = $p THEN svpt_1 ELSE svpt_2 END) AS avg_svpt,
      AVG(CASE WHEN ID_1 = $p THEN bpSaved_1 ELSE bpSaved_2 END) AS avg_bpSaved
    FROM matches
    WHERE {" AND ".join(where)}
    """
    df = sql_df(q, params)
    if df.empty:
        return 0, 0, 0
    return df.iloc[0]["avg_aces"], df.iloc[0]["avg_svpt"], df.iloc[0]["avg_bpSaved"]

def wl_grouped(p_id, start_date, end_date, level, group_col):
    where, params = build_where_player_only(p_id, start_date, end_date, level)

    if group_col not in COLS and group_col != "year":
        return pd.DataFrame(columns=[group_col, "result", "cnt"])

    group_expr = "EXTRACT(year FROM CAST(tournament_date AS DATE))::INT" if group_col == "year" else group_col

    q = f"""
    SELECT
      {group_expr} AS g,
      CASE
        WHEN (ID_1 = $p AND Winner = 0) OR (ID_2 = $p AND Winner = 1) THEN 'Win'
        ELSE 'Loss'
      END AS result,
      COUNT(*) AS cnt
    FROM matches
    WHERE {where}
      AND {group_expr} IS NOT NULL
    GROUP BY g, result
    ORDER BY g
    """
    df = sql_df(q, params)
    df = df.rename(columns={"g": group_col})
    return df

def top_opponents(p_id, start_date, end_date, level, limit=50):
    where, params = build_where_player_only(p_id, start_date, end_date, level)

    q = f"""
    WITH base AS (
      SELECT
        CASE WHEN ID_1 = $p THEN ID_2 ELSE ID_1 END AS opp_id,
        CASE WHEN ID_1 = $p THEN Name_2 ELSE Name_1 END AS opp_name,
        CASE
          WHEN (ID_1 = $p AND Winner = 0) OR (ID_2 = $p AND Winner = 1) THEN 1.0
          ELSE 0.0
        END AS is_win
      FROM matches
      WHERE {where}
        AND (CASE WHEN ID_1 = $p THEN ID_2 ELSE ID_1 END) IS NOT NULL
    )
    SELECT
      opp_name,
      opp_id,
      COUNT(*) AS matches,
      AVG(is_win) AS win_rate
    FROM base
    GROUP BY opp_name, opp_id
    HAVING COUNT(*) >= 3
    ORDER BY matches DESC
    LIMIT {int(limit)}
    """
    df = sql_df(q, params)
    if not df.empty:
        df["win_rate"] = (df["win_rate"] * 100).round(1).astype(str) + "%"
    return df

def recent_matches_player(p_id, start_date, end_date, level, limit=30):
    where, params = build_where_player_only(p_id, start_date, end_date, level)

    cols = [c for c in ["tournament_date","tournament","tournament_surface","tournament_level","round","score","ID_1","Name_1","ID_2","Name_2","Winner"] if c in COLS]
    q = f"""
    SELECT {", ".join(cols)}
    FROM matches
    WHERE {where}
    ORDER BY CAST(tournament_date AS DATE) DESC
    LIMIT {int(limit)}
    """
    df = sql_df(q, params)
    if df.empty:
        return df

    df["Result"] = np.where(
        ((df["ID_1"] == p_id) & (df["Winner"] == 0)) | ((df["ID_2"] == p_id) & (df["Winner"] == 1)),
        "Win", "Loss"
    )
    df["tournament_date"] = pd.to_datetime(df["tournament_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df

def win_round_counts(pid, start_date=None, end_date=None, level="(All)"):
    where = ["(ID_1 = $p OR ID_2 = $p)"]
    params = {"p": pid}

    if start_date and end_date:
        where.append("CAST(tournament_date AS DATE) BETWEEN $start AND $end")
        params["start"] = str(start_date)
        params["end"] = str(end_date)

    if level and level != "(All)" and "tournament_level" in COLS:
        where.append("tournament_level = $lvl")
        params["lvl"] = level

    round_col = "round"

    q = f"""
    SELECT
      CASE
        WHEN (ID_1 = $p AND Winner = 0) OR (ID_2 = $p AND Winner = 1) THEN 'Win'
        ELSE 'Loss'
      END AS result,
      {round_col} AS round,
      COUNT(*) AS cnt
    FROM matches
    WHERE {" AND ".join(where)}
      AND {round_col} IS NOT NULL
    GROUP BY result, round
    """
    return sql_df(q, params)