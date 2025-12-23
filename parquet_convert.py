import glob
import os
import pandas as pd

SRC_GLOB = r"data_tennis_match_reduced/matches_data_*.csv"
OUT_DIR  = r"data_tennis_match_reduced/parquet"

os.makedirs(OUT_DIR, exist_ok=True)

for fp in glob.glob(SRC_GLOB):
    year = os.path.splitext(os.path.basename(fp))[0].split("_")[-1]  # gets 2024 from matches_data_2024
    out_fp = os.path.join(OUT_DIR, f"matches_{year}.parquet")

    df = pd.read_csv(fp, sep=";", engine="python")
    # parse date once (important)
    if "tournament_date" in df.columns:
        df["tournament_date"] = pd.to_datetime(df["tournament_date"], errors="coerce")

    df.to_parquet(out_fp, index=False)   # requires pyarrow or fastparquet
    print("Wrote", out_fp)