import argparse
import gzip
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert HSL HFP MQTT data dump files to GeoParquet format.")
    parser.add_argument("--rt-files", nargs="+", required=True, help="List of HFP data files (text or gzip).")
    parser.add_argument(
        "--bbox",
        default="25.007142,60.221397,25.023972,60.229863",
        help="Bounding box for filtering (lon_min,lat_min,lon_max,lat_max).",
    )
    parser.add_argument("--dir", type=str, help='Direction to filter (e.g., "1" or "2").')
    parser.add_argument("--outfile", required=True, nargs="+", help="Output file (CSV or Parquet based on extension).")
    parser.add_argument("--log", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    return args


def parse_hfp_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single HFP data line.

    Returns None if the line doesn't contain VP data or is malformed.
    """
    try:
        # Split the MQTT topic and the message payload
        # Find the first { character which indicates start of JSON payload
        json_start = line.find("{")
        if json_start == -1:
            return None
        # Split into topic and payload based on JSON start position
        topic = line[:json_start].strip()
        payload = line[json_start:].strip()
        # Only process VP (vehicle position) messages
        if "/vp/" not in topic:
            return None

        # Parse the JSON payload
        data = json.loads(payload)
        # Only process VP data (double check)
        if "VP" not in data:
            return None

        # Return the VP data directly, which contains all the fields we need
        return data["VP"]
    except Exception:
        # Skip malformed lines
        return None


def is_within_bbox(lon: float, lat: float, bbox: Tuple[float, float, float, float]) -> bool:
    """Check if the coordinates are within the bounding box."""
    lon_min, lat_min, lon_max, lat_max = bbox
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


def process_file(
    file_path: str, bbox: Tuple[float, float, float, float], direction: Optional[str]
) -> List[Dict[str, Any]]:
    """
    Process a single HFP data file with filtering.

    Args:
        file_path: Path to the HFP data file
        bbox: Bounding box as (lon_min, lat_min, lon_max, lat_max)
        direction: Direction to filter (None to include all)

    Returns:
        List of parsed records that match the filter criteria
    """
    results = []

    # Check if the file is gzipped
    open_func, mode = (gzip.open, "rt") if file_path.endswith(".gz") else (open, "r")

    # Process the file with a progress bar
    with open_func(file_path, mode) as f:
        for line in tqdm(f, desc=f"Processing {Path(file_path).name}", unit="lines", total=None):
            record = parse_hfp_line(line)
            if record is None:
                continue

            # Apply bbox filter
            lon, lat = record.get("long"), record.get("lat")
            if lon is None or lat is None or not is_within_bbox(lon, lat, bbox):
                continue

            # Apply direction filter if specified
            if direction is not None and record.get("dir") != direction:
                continue

            results.append(record)

    return results


def process_files(
    file_paths: List[str], bbox: Tuple[float, float, float, float], direction: Optional[str], chunk_size: int = 100000
) -> pd.DataFrame:
    """
    Process multiple HFP files in chunks to avoid loading everything into memory.

    Args:
        file_paths: List of HFP data files
        bbox: Bounding box for filtering
        direction: Direction to filter (None to include all)
        chunk_size: Number of records to process before creating a DataFrame

    Returns:
        DataFrame with all the processed records
    """
    all_dfs = []
    current_chunk = []

    for file_path in file_paths:
        print(f"Processing file: {file_path}")

        # Process each file
        records = process_file(file_path, bbox, direction)

        # Add records to the current chunk
        current_chunk.extend(records)

        # If the chunk is large enough, convert to DataFrame and add to the list
        if len(current_chunk) >= chunk_size:
            df_chunk = pd.DataFrame(current_chunk)
            all_dfs.append(df_chunk)
            current_chunk = []  # Reset the chunk

    # Process the remaining records
    if current_chunk:
        df_chunk = pd.DataFrame(current_chunk)
        all_dfs.append(df_chunk)

    # Combine all DataFrames
    if not all_dfs:
        return pd.DataFrame()  # Return empty DataFrame if no data
    df = pd.concat(all_dfs, ignore_index=True)
    print(df.head())
    return df


def save_dataframe(df: pd.DataFrame, outfile: List[str]):
    """Save the DataFrame to the specified output file(s)."""
    print(f"Saving data to {outfile}")

    # Ensure the output directory exists
    for of in outfile:
        os.makedirs(os.path.dirname(os.path.abspath(of)), exist_ok=True)

    # Save based on file extension
    for of in outfile:
        if of.endswith(".geoparquet"):
            # Create geometry column from lat/long
            geometry = [Point(xy) for xy in zip(df["long"], df["lat"])]

            # Convert to GeoDataFrame, in WGS84 coordinate system
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

            # Save as GeoParquet
            gdf.to_parquet(of, index=False)

        elif of.endswith(".parquet"):
            df.to_parquet(of)
        elif of.endswith(".csv"):
            df.to_csv(of, index=False)
        else:
            raise ValueError("Output file must have .parquet, .geoparquet, or .csv extension")

    print(f"Data saved successfully. Total records: {len(df)}")


def main():
    args = parse_args()

    # Parse the bounding box
    bbox = tuple(map(float, args.bbox.split(",")))
    if len(bbox) != 4:
        raise ValueError("Bounding box must have format: lon_min,lat_min,lon_max,lat_max")

    # Process all files and create the DataFrame
    df = process_files(args.rt_files, bbox, args.dir)

    # Calculate some additional columns for analysis
    if not df.empty:
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["tst"])

        # Sort by vehicle and timestamp
        df = df.sort_values(["veh", "timestamp"])

        # Calculate speed changes (for deceleration detection)
        df["speed_diff"] = df.groupby("veh")["spd"].diff()

        # Calculate time diff in seconds
        df["time_diff"] = df.groupby("veh")["tsi"].diff()

        # Calculate acceleration explicitly (change in speed / change in time)
        # This can be compared with the 'acc' value reported in the data
        df["calc_acc"] = df["speed_diff"] / df["time_diff"]

        # Flag potential sudden braking events
        # (this is just a simple heuristic, can be refined in analysis)
        df["potential_braking"] = (df["speed_diff"] < -1.5) & (df["time_diff"] < 3)

    # Save the DataFrame
    save_dataframe(df, args.outfile)


if __name__ == "__main__":
    main()
