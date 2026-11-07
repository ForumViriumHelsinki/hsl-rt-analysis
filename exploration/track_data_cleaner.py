"""
Script for cleaning and processing HSL vehicle GPS track data.

This script reads GPS data from parquet files, applies various filters,
creates track segments from vehicle movements, and optionally filters
track segments based on geographic buffers around predefined track lines.

The cleaned data is saved as a parquet file for use in visualization
or further analysis.

--- SAMPLE INPUT DATA (random rows) ---
       desi  dir  oper  veh   spd  hdg   acc    dl    odo   drst   jrn  line  loc     stop   route  occu                               ts          oday_start                   geometry
419923   15    1    40  618  3.68  101 -0.86   -75  23146  False   133  1142  GPS  2222403    2015     0 2025-07-31 10:47:20.755000+00:00 2025-07-31 12:52:00  POINT (24.81167 60.18737)
743323   15    2    40  613  5.55  358  0.08    23   5741  False   700  1142  GPS     <NA>    2015     0 2025-07-31 17:25:21.251000+00:00 2025-07-31 20:13:00  POINT (24.81011 60.21341)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Polygon, box, shape


def load_geometry_feature(config_path: str, feature_name: str) -> tuple[Union[LineString, Polygon], dict]:
    """Load geometry feature from GeoJSON configuration file.

    Args:
        config_path: Path to GeoJSON configuration file
        feature_name: Name of the feature to load

    Returns:
        Tuple of (shapely geometry object, properties dictionary)

    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If feature_name is not found in config
        ValueError: If geometry type is not supported
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"GeoJSON configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        geojson_data = json.load(f)

    # Find the feature by name
    feature = None
    available_features = []

    for feat in geojson_data.get("features", []):
        feature_props = feat.get("properties", {})
        name = feature_props.get("name")
        available_features.append(name)

        if name == feature_name:
            feature = feat
            break

    if feature is None:
        raise KeyError(f"Feature '{feature_name}' not found in GeoJSON. Available features: {available_features}")

    # Convert GeoJSON geometry to shapely object
    geometry = shape(feature["geometry"])
    properties = feature.get("properties", {})

    # Validate geometry type
    if not isinstance(geometry, (LineString, Polygon)):
        raise ValueError(
            f"Unsupported geometry type: {type(geometry).__name__}. Only LineString and Polygon are supported."
        )

    return geometry, properties


def create_geometry_buffer(geometry: Union[LineString, Polygon], buffer_size_meters: float) -> Polygon:
    """Create a buffer polygon around the given geometry.

    Args:
        geometry: Shapely LineString or Polygon geometry
        buffer_size_meters: Buffer size in meters

    Returns:
        Polygon representing the buffered area
    """
    # Create buffer - note: this is in degrees, approximate conversion
    # 1 degree of latitude ≈ 111,000 meters
    # For longitude at Helsinki (60°N), 1 degree ≈ 55,500 meters
    buffer_degrees = buffer_size_meters / 111000  # Simple approximation

    buffered_geometry = geometry.buffer(buffer_degrees)

    # Ensure we always return a Polygon (buffer of LineString is Polygon, buffer of Polygon is Polygon)
    if isinstance(buffered_geometry, Polygon):
        return buffered_geometry
    else:
        # Handle edge cases where buffer might return other geometry types
        raise ValueError(f"Buffer operation resulted in unexpected geometry type: {type(buffered_geometry).__name__}")


def filter_tracks_by_buffer(tracks: dict, buffer_polygon: Polygon) -> dict:
    """Filter track segments to only include those completely within buffer.

    Args:
        tracks: Dictionary of vehicle tracks
        buffer_polygon: Buffer polygon to filter against

    Returns:
        Filtered tracks dictionary
    """
    filtered_tracks = {}

    for veh, vehicle_trips in tracks.items():
        filtered_trips = []

        for trip in vehicle_trips:
            # Check if the entire linestring is within the buffer
            if buffer_polygon.contains(trip["linestring"]):
                filtered_trips.append(trip)

        # Only keep vehicles that have at least one valid trip
        if filtered_trips:
            filtered_tracks[veh] = filtered_trips

    return filtered_tracks


def filter_dataframe_by_tracks(df: gpd.GeoDataFrame, tracks: dict) -> gpd.GeoDataFrame:
    """Filter dataframe to only include rows that correspond to points in the tracks.

    Args:
        df: Original GeoDataFrame
        tracks: Dictionary of vehicle tracks (after potential filtering)

    Returns:
        Filtered GeoDataFrame containing only points that are in the tracks
    """
    if not tracks:
        return df.iloc[0:0].copy()  # Return empty dataframe with same structure

    # Collect all track coordinates and their properties
    track_points = []

    for veh, vehicle_trips in tracks.items():
        for trip in vehicle_trips:
            coords = list(trip["linestring"].coords)
            # Add coordinate info with vehicle and trip metadata
            for coord in coords:
                track_points.append(
                    {
                        "lon": coord[0],
                        "lat": coord[1],
                        "veh": veh,
                        "first_time": trip["first_time"],
                        "last_time": trip["last_time"],
                    }
                )

    if not track_points:
        return df.iloc[0:0].copy()

    # Create a set of coordinates for faster lookup
    track_coords = {(round(pt["lon"], 6), round(pt["lat"], 6)) for pt in track_points}

    # Filter original dataframe to match track points
    # We need to match by coordinates and vehicle number for precision
    mask = df.apply(
        lambda row: (round(row.geometry.x, 6), round(row.geometry.y, 6)) in track_coords and row["veh"] in tracks,
        axis=1,
    )

    return df[mask].copy()


def create_vehicle_tracks(df: gpd.GeoDataFrame, time_threshold_seconds: int = 5) -> tuple[dict, gpd.GeoDataFrame]:
    """Create LineString tracks for each vehicle from GPS points, splitting trips based on time gaps.

    Args:
        df: GeoDataFrame with vehicle GPS data
        time_threshold_seconds: Maximum time gap in seconds between consecutive points
                               before splitting into separate trips (default: 5)

    Returns:
        Tuple containing:
        - Dictionary with vehicle numbers as keys and lists of trip dictionaries as values.
          Each trip contains linestring, timestamps, and point count information.
        - GeoDataFrame containing only the rows that are included in the tracks
    """
    tracks = {}
    used_indices = []  # Keep track of all indices used in tracks

    for veh in df["veh"].unique():
        veh_data = df[df["veh"] == veh].copy()

        # Sort by timestamp to ensure correct order
        veh_data = veh_data.sort_values("ts")

        # Store original indices before reset
        original_indices = veh_data.index.tolist()

        # Reset index for easier iteration
        veh_data = veh_data.reset_index(drop=True)

        # Split into trips based on time gaps
        trips = []
        current_trip_indices = [0]  # Start with first point

        for i in range(1, len(veh_data)):
            # Calculate time difference in seconds
            time_diff = (veh_data["ts"].iloc[i] - veh_data["ts"].iloc[i - 1]).total_seconds()

            if time_diff > time_threshold_seconds:
                # Time gap detected - end current trip and start new one
                if len(current_trip_indices) >= 2:  # Need at least 2 points for a line
                    trips.append(current_trip_indices)
                current_trip_indices = [i]  # Start new trip
            else:
                # Continue current trip
                current_trip_indices.append(i)

        # Add the last trip if it has enough points
        if len(current_trip_indices) >= 2:
            trips.append(current_trip_indices)

        # Create LineString for each trip
        vehicle_trips = []
        for trip_idx, trip_indices in enumerate(trips):
            trip_data = veh_data.iloc[trip_indices]

            # Create list of coordinates (lon, lat)
            coords = [(point.x, point.y) for point in trip_data.geometry]

            if len(coords) >= 2:  # Double check we have enough points
                # Get first and last timestamps for this trip
                first_time = trip_data["ts"].iloc[0]
                last_time = trip_data["ts"].iloc[-1]

                trip_info = {
                    "linestring": LineString(coords),
                    "first_time": first_time,
                    "last_time": last_time,
                    "trip_id": trip_idx,
                    "point_count": len(coords),
                    "vehicle_id": veh,
                }
                vehicle_trips.append(trip_info)

                # Add the original indices of points used in this trip
                used_indices.extend([original_indices[i] for i in trip_indices])

        if vehicle_trips:  # Only add if we have valid trips
            tracks[veh] = vehicle_trips

    # Create filtered dataframe with only used indices
    filtered_df = df.loc[used_indices].copy() if used_indices else df.iloc[0:0].copy()  # Empty df with same structure

    return tracks, filtered_df


def filter_dataframe(df: gpd.GeoDataFrame, args: argparse.Namespace) -> gpd.GeoDataFrame:
    """Apply basic filters to the dataframe.

    Args:
        df: Input GeoDataFrame
        args: Command line arguments

    Returns:
        Filtered GeoDataFrame
    """
    logging.info(f"Original dataset size: {len(df)} rows")

    # Drop rows where location is not defined (geometry is None/NaN)
    df = df.dropna(subset=["geometry"])
    logging.info(f"After removing rows with undefined location: {len(df)} rows")

    # Apply bounding box filter
    if args.bbox:
        bbox = [float(coord) for coord in args.bbox.split(",")]
        df = df[df.geometry.within(box(bbox[0], bbox[1], bbox[2], bbox[3]))]
        logging.info(f"After bounding box filter ({bbox}): {len(df)} rows")

    # Apply vehicle filter
    if args.veh:
        df = df[df["veh"].isin(args.veh)]
        logging.info(f"After vehicle filter: {len(df)} rows")

    # Apply direction filter
    if args.dir:
        df = df[df["dir"] == args.dir]
        logging.info(f"After direction filter: {len(df)} rows")

    # Apply row limit
    if args.limit_rows and args.limit_rows > 0:
        df = df.head(args.limit_rows)
        logging.info(f"After row limit: {len(df)} rows")

    return df


def read_parquet_files(paths: list[str]) -> gpd.GeoDataFrame:
    """Read and combine multiple parquet files.

    Args:
        paths: List of paths to parquet files

    Returns:
        Combined GeoDataFrame
    """
    dfs = []
    for path in paths:
        logging.info(f"Reading {path}")
        dfs.append(gpd.read_parquet(path))
    return pd.concat(dfs, ignore_index=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Clean and process HSL vehicle GPS track data")

    # Input/Output
    parser.add_argument("--input", type=str, nargs="+", required=True, help="Path(s) to the input GeoParquet file(s)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the cleaned data as parquet file")

    # Basic filters
    parser.add_argument("--limit-rows", type=int, help="Limit the number of rows to process")
    parser.add_argument("--bbox", type=str, help="Bounding box in format 'lon_min,lat_min,lon_max,lat_max'")
    parser.add_argument("--veh", type=int, nargs="+", help="Filter data for specific vehicle number(s)")
    parser.add_argument("--dir", type=int, help="Filter data for specific direction number")

    # Track processing
    parser.add_argument(
        "--time-threshold",
        type=int,
        default=3,
        help="Time threshold in seconds for splitting vehicle trips (default: 3)",
    )

    # Buffer filtering
    parser.add_argument(
        "--geometry-config",
        type=str,
        default="track_geometries.geojson",
        help="Path to GeoJSON geometry configuration file (default: track_geometries.geojson)",
    )
    parser.add_argument("--feature-name", type=str, help="Name of the geometry feature to use from GeoJSON file")
    parser.add_argument(
        "--buffer-size", type=float, help="Buffer size in meters around geometry for filtering track segments"
    )

    # Logging
    parser.add_argument(
        "--log", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=args.log, format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Validate buffer-related arguments
    if args.buffer_size and not args.feature_name:
        parser.error("--feature-name is required when using --buffer-size")

    return args


def main():
    """Main processing function."""
    args = parse_args()

    # Read input data
    df = read_parquet_files(args.input)

    # Apply basic filters
    df = filter_dataframe(df, args)

    if df.empty:
        logging.warning("No data remaining after filters. Exiting.")
        return

    # Create vehicle tracks
    logging.info(f"Creating tracks for {len(df['veh'].unique())} vehicles with {args.time_threshold}s time threshold")
    tracks, filtered_df = create_vehicle_tracks(df, time_threshold_seconds=args.time_threshold)

    if not tracks:
        logging.warning("No tracks were created. Check your data and filters.")
        return

    # Apply buffer filtering if specified
    if args.buffer_size and args.feature_name:
        logging.info(f"Loading geometry feature '{args.feature_name}' from {args.geometry_config}")
        try:
            geometry, properties = load_geometry_feature(args.geometry_config, args.feature_name)
            geometry_type = type(geometry).__name__

            # Check if feature has default buffer size in properties
            default_buffer = properties.get("buffer_meters", 0)
            actual_buffer = args.buffer_size if args.buffer_size > 0 else default_buffer

            if actual_buffer > 0:
                logging.info(f"Applying {actual_buffer}m buffer to {geometry_type} geometry '{args.feature_name}'")
                buffer_polygon = create_geometry_buffer(geometry, actual_buffer)
            else:
                logging.info(f"Using {geometry_type} geometry '{args.feature_name}' without buffer")
                if isinstance(geometry, Polygon):
                    buffer_polygon = geometry
                else:
                    raise ValueError("Cannot use LineString geometry without buffer. Specify --buffer-size > 0")

            original_trip_count = sum(len(vehicle_trips) for vehicle_trips in tracks.values())
            tracks = filter_tracks_by_buffer(tracks, buffer_polygon)
            filtered_trip_count = sum(len(vehicle_trips) for vehicle_trips in tracks.values())

            logging.info(f"Geometry filtering: {original_trip_count} -> {filtered_trip_count} trips remaining")

            # Update filtered dataframe to match the filtered tracks
            if tracks:
                filtered_df = filter_dataframe_by_tracks(filtered_df, tracks)
                logging.info(f"Filtered dataframe has {len(filtered_df)} rows after geometry filtering")
            else:
                logging.warning("No tracks remaining after geometry filtering")
                filtered_df = filtered_df.iloc[0:0].copy()  # Empty dataframe

        except (FileNotFoundError, KeyError, ValueError) as e:
            logging.error(f"Error loading geometry configuration: {e}")
            return

    # Save cleaned data
    if not filtered_df.empty:
        logging.info(f"Saving cleaned data to {args.output}")
        filtered_df.to_parquet(args.output)

        # Summary statistics
        total_trips = sum(len(vehicle_trips) for vehicle_trips in tracks.values())
        unique_vehicles = len(tracks)

        logging.info("Data cleaning complete:")
        logging.info(f"  - {len(filtered_df)} GPS points saved")
        logging.info(f"  - {unique_vehicles} vehicles")
        logging.info(f"  - {total_trips} track segments")
        logging.info(f"  - Saved to: {args.output}")
    else:
        logging.warning("No data to save - filtered dataframe is empty")


if __name__ == "__main__":
    main()
