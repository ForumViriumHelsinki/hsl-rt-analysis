import argparse
import pathlib
import sys
from datetime import datetime

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq


def analyze_parquet(file_path):
    """Analyze parquet file and display key statistics, including spatial properties if available."""

    # Check if file exists
    if not pathlib.Path(file_path).exists():
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    try:
        # Read parquet file
        print(f"\n{'=' * 80}")
        print(f"PARQUET FILE ANALYSIS: {pathlib.Path(file_path).name}")
        print(f"{'=' * 80}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"File path: {pathlib.Path(file_path).absolute()}")
        print(f"File size: {pathlib.Path(file_path).stat().st_size / (1024 * 1024):.2f} MB")

        # Read parquet file metadata first
        parquet_file = pq.ParquetFile(file_path)
        num_row_groups = parquet_file.num_row_groups
        total_rows = 0
        for i in range(num_row_groups):
            total_rows += parquet_file.metadata.row_group(i).num_rows

        print("\n--- PARQUET-METADATA ---")
        print(f"Row count:  {total_rows}")
        print(f"Row groups: {num_row_groups}")
        print(f"Columns:    {len(parquet_file.schema_arrow)}")

        # Try to read as GeoParquet first
        try:
            df = gpd.read_parquet(file_path)
            is_geo = True
        except (gpd.io.file.DriverError, ValueError):
            # DriverError occurs when file is not a valid GeoParquet
            # ValueError occurs when geometry column is missing or invalid
            df = pd.read_parquet(file_path)
            is_geo = False

        print("\n--- DATAFRAME OVERVIEW ---")
        print(f"Number of rows:    {df.shape[0]}")
        print(f"Number of columns: {df.shape[1]}")

        # If it's a GeoDataFrame, show spatial information
        if is_geo:
            print("\n--- SPATIAL PROPERTIES ---")
            print(f"Geometry type: {df.geometry.geom_type.value_counts().to_dict()}")
            print(f"CRS: {df.crs}")
            bounds = df.total_bounds
            print("Spatial extent:")
            print(f"  - X min, max:   {bounds[0]:.6f}, {bounds[2]:.6f}")
            print(f"  - Y min, max:   {bounds[1]:.6f}, {bounds[3]:.6f}")
            if df.geometry.has_z.any():
                z_min = df.geometry.z.min()
                z_max = df.geometry.z.max()
                print(f"  - Z min, max: {z_min:.6f}, {z_max:.6f}")
            print(f"  - Bounding box: {bounds[0]:.6f},{bounds[1]:.6f},{bounds[2]:.6f},{bounds[3]:.6f}")

        # Show data types
        print("\n--- COLUMN TYPES ---")
        dtypes_summary = df.dtypes.value_counts().to_dict()
        for dtype, count in dtypes_summary.items():
            print(f"{dtype}: {count} columns")

        # Show first rows
        print("\n--- SAMPLE DATA (first 5 rows) ---")
        print(df.head().to_string())

        # Calculate missing values
        print("\n--- MISSING VALUES ---")
        null_counts = df.isna().sum()
        if null_counts.sum() > 0:
            nulls = null_counts[null_counts > 0]
            for col, count in nulls.items():
                print(f"{col}: {count} missing values ({count / len(df) * 100:.2f}%)")
        else:
            print("No missing values.")

        # Calculate statistics for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            print("\n--- NUMERIC STATISTICS ---")
            print(df[numeric_cols].describe().transpose().to_string())

        # Calculate string column lengths
        string_cols = df.select_dtypes(include=["object", "string"]).columns
        if len(string_cols) > 0:
            print("\n--- STRING COLUMN LENGTHS ---")
            for col in string_cols:
                if df[col].dropna().empty:
                    continue
                lengths = df[col].astype(str).str.len()
                print(f"{col}: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.2f}")

        # Categorical columns (max 10 most common values)
        cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns
        if len(cat_cols) > 0:
            print("\n--- CATEGORICAL COLUMN MOST COMMON VALUES ---")
            for col in cat_cols:
                if df[col].dropna().empty:
                    continue
                value_counts = df[col].value_counts().head(5)
                unique_count = df[col].nunique()
                print(f"{col} (unique values: {unique_count}):")
                for val, count in value_counts.items():
                    print(f"  - {val}: {count} ({count / len(df) * 100:.2f}%)")

        print(f"\n{'=' * 80}")
        print("Analysis complete.")

    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze Parquet file contents and display statistics, including spatial properties for GeoParquet files."
    )
    parser.add_argument("file_path", help="Path to the Parquet file")

    args = parser.parse_args()
    analyze_parquet(args.file_path)
