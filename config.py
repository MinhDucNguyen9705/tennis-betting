PARQUET_GLOB = r"D:/OneDrive - Hanoi University of Science and Technology/Domain Basics/Introduction to Business Analytics/data_tennis_match_reduced/parquet/matches_*.parquet"

LATEST_SKILLS = {
    "Aces_Percentage",
    "First_Serve_Success_Percentage",
    "Winning_on_1st_Serve_Percentage",
    "Winning_on_2nd_Serve_Percentage",
    "BreakPoint_Saved_Percentage",
    "Overall_Win_on_Serve_Percentage",
}

MEAN_SKILLS = {"minutes_fatigue"}
FORM_SKILL = "Victories_Percentage"

SKILLS = {
    "Aces_Percentage": "Aces",
    "First_Serve_Success_Percentage": "1st Serve In",
    "Winning_on_1st_Serve_Percentage": "1st Serve Won",
    "Winning_on_2nd_Serve_Percentage": "2nd Serve Won",
    "Overall_Win_on_Serve_Percentage": "Overall Serve Won",
    "BreakPoint_Saved_Percentage": "BP Saved",
    "Victories_Percentage": "Win Rate Last 10 Matches",
    "minutes_fatigue": "Stamina (min)",
}

REQUIRED_BASE = {
    "ID_1","ID_2","Name_1","Name_2",
    "tournament_date","tournament_surface","tournament_level","Winner"
}

DUCKDB_THREADS = 8