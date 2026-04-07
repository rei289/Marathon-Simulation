"""Test file."""
import pandas as pd

if __name__ == "__main__":
    # retrieve parquet run data
    df = pd.read_parquet("running_simulation_data/01_runs/2026-04-06_13-50/2026-04-06_13-50_streams.parquet", engine="pyarrow")
    print(df.head())
