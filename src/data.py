import pandas as pd
from pathlib import Path
from .config import RAW_DIR, PROCESSED_DIR, DECIMAL_PRECISION

file_path = RAW_DIR / "taguchi_datasets.csv"

#load taguchi parameters from csv
def load_taguchi_csv(filename="taguchi_datasets.csv")->pd.DataFrame:
    filepath = RAW_DIR / filename
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    return df

#load 27 force signals from csv files (TG1-TG27)
def load_force_signals(prefix="TG", n_runs: int=27)->dict[int, pd.DataFram]:
    signals = {}
    for i in range(1, n_runs+1):
        filename = RAW_DIR / f"{prefix}{i}.csv"
        if not filename.exists():
            print(f"Warning: File {filename} does not exist. Skipping.")
            continue

        df = pd.read_csv(file_path)
        df.columns = [c.strip() for c in df.columns]

        #ensure required columns exist
        required_cols={"time","fx","fy","fz"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"File {file_path.name} missing columns: {required_cols - set(df.columns)}")
        
        df["run_id"] = i
        signals[i] = df
    return signals
#merge taguchi parameters with force signals
def merge_experiment_data() -> pd.DataFrame:
    params = load_taguchi_csv()
    signals = load_force_signals()

    summary = []
    for i, df in signals.items():
        row = {
            "run_id": i,
            "Fx_mean": df["Fx"].mean(),
            "Fy_mean": df["Fy"].mean(),
            "Fz_mean": df["Fz"].mean(),
            "Fx_std": df["Fx"].std(),
            "Fy_std": df["Fy"].std(),
            "Fz_std": df["Fz"].std(),
        }
        summary.append(row)

    df_summary = pd.DataFrame(summary)
    merged = pd.merge(params, df_summary, on="run_id", how="inner")

    # Save merged file for convenience
    out_path = PROCESSED_DIR / "taguchi_summary.csv"
    merged.to_csv(out_path, index=False, float_format=f"%.{DECIMAL_PRECISION}f", encoding="utf-8")

    print(f"Saved merged dataset to: {out_path}")
    return merged

#main
if __name__ == "__main__":
    print("Loading Taguchi parameters...")
    params = load_taguchi_csv()
    print(params.head())

    print("\nLoading all TG force files...")
    signals = load_force_signals()
    print(f"Loaded {len(signals)} runs.")

    print("\nMerging parameters and summary...")
    merged_df = merge_experiment_data()
    print(merged_df.head())