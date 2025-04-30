"""
This script is used to process HFP data from the HSL HFP MQTT data dump files to GeoParquet format.

Sample lines in raw data files:

/hfp/v2/journey/ongoing/vp/tram/0040/00436/1003/1/Pikku Huopalahti/12:50/1160403/4/60;24/28/09/38 {"VP": {"desi": "3", "dir": "1", "oper": 40, "veh": 436, "tst": "2025-03-12T11:24:51.251Z", "tsi": 1741778691, "spd": 5.5, "hdg": 268, "lat": 60.203263, "long": 24.898389, "acc": 0.11, "dl": 262, "odo": 9070, "drst": 0, "oday": "2025-03-12", "jrn": 762, "line": 31, "start": "12:50", "loc": "GPS", "stop": null, "route": "1003", "occu": 0}}
/hfp/v2/journey/ongoing/vp/bus/0022/01400/2212/1/Kauniala/11:26/2252204/5/60;24/27/08/15 {"VP": {"desi": "212", "dir": "1", "oper": 22, "veh": 1400, "tst": "2025-03-12T10:00:04.751Z", "tsi": 1741773604, "spd": 1.08, "hdg": 268, "lat": 60.201181, "long": 24.785763, "acc": 1.08, "dl": -120, "odo": 12579, "drst": 0, "oday": "2025-03-12", "jrn": 525, "line": 244, "start": "11:26", "loc": "GPS", "stop": 2252204, "route": "2212", "occu": 0}}
/hfp/v2/journey/ongoing/vp/bus/0018/01095/1065/1/Veräjälaakso/11:55/1111117/5/60;24/19/75/71 {"VP": {"desi": "65", "dir": "1", "oper": 6, "veh": 1095, "tst": "2025-03-12T10:00:02.751Z", "tsi": 1741773602, "spd": 7.4, "hdg": 354, "lat": 60.177785, "long": 24.951814, "acc": -0.38, "dl": -70, "odo": 1139, "drst": 0, "oday": "2025-03-12", "jrn": 1089, "line": 850, "start": "11:55", "loc": "GPS", "stop": null, "route": "1065", "occu": 0}}

Only VP (vehicle position) messages are stored in the parquet files.
All other messages are stored in the compressed text files next to the parquet files.

Source files are named using the following naming convention:

<route>-<year>-<month>-<day>T<hour>.txt[.gz]
e.g. 9993K-2025-03-14T13.txt.gz

where:
- <route> is the route number (e.g. 9993, sometimes trailing characters are present)
- <year> is the 4-digit year (e.g. 2025)
- <month> is the 2-digit month (e.g. 03)
- <day> is the 2-digit day (e.g. 14)
- <hour> is the 2-digit hour in 24h format (e.g. 13)
- .gz extension is present if file is gzipped

One output file per UTC day.

Routes from 1001 to 4999 are stored one file per route.

Routes from 5000 to 5999, 6000 to 6999, 7000 to 7999, 9000 to 9999 are are aggregated into a single files.

Files are stored in the output directory with the following naming convention:

YYYY-MM-DD/<route>_YYYYMMDD.geoparquet

where <year> is the year, <month> is the month, <day> is the day and <route> is the first 4 digits of the route number.

Data field descriptions (from Digitransit API documentation):
https://digitransit.fi/en/developers/apis/5-realtime-api/vehicle-positions/high-frequency-positioning/#the-payload

Field   Type    Description
desi    str     Route number visible to passengers.
dir     str     Route direction of the trip. After type conversion matches direction_id in GTFS and the topic. Either "1" or "2".
oper    int     Unique ID of the operator running the trip (i.e. this value can be different than the operator ID in the topic,
                for example if the service has been subcontracted to another operator). The unique ID does not have prefix zeroes here.
veh     int     Vehicle number that can be seen painted on the side of the vehicle, often next to the front door.
                Different operators may use overlapping vehicle numbers. Matches vehicle_number in the topic except without the prefix zeroes.
tst     str     UTC timestamp with millisecond precision from the vehicle in ISO 8601 format (yyyy-MM-dd'T'HH:mm:ss.SSSZ).
tsi     int     Unix time in seconds from the vehicle.
spd     float   Speed of the vehicle, in meters per second (m/s).
hdg     int     Heading of the vehicle, in degrees (⁰) starting clockwise from geographic north. Valid values are on the closed interval [0, 360].
lat     float   WGS 84 latitude in degrees. null if location is unavailable.
long    float   WGS 84 longitude in degrees. null if location is unavailable.
acc     float   Acceleration (m/s^2), calculated from the speed on this and the previous message.
                Negative values indicate that the speed of the vehicle is decreasing.
dl      int     Offset from the scheduled timetable in seconds (s). Negative values indicate lagging behind the schedule,
                positive values running ahead of schedule.
odo     int     The odometer reading in meters (m) since the start of the trip. Currently the values not very reliable.
drst    int     Door status. 0 if all the doors are closed. 1 if any of the doors are open.
oday    str     Operating day of the trip. The exact time when an operating day ends depends on the route. For most routes,
                the operating day ends at 4:30 AM on the next day. In that case, for example, the final moment of the operating day "2018-04-05"
                would be at 2018-04-06T04:30 local time.
jrn     int     Internal journey descriptor, not meant to be useful for external use.
line    int     Internal line descriptor, not meant to be useful for external use.
start   str     Scheduled start time of the trip, i.e. the scheduled departure time from the first stop of the trip. The format follows
                HH:mm in 24-hour local time, not the 30-hour overlapping operating days present in GTFS. Matches start_time in the topic.
loc     str     Location source, either GPS, ODO, MAN, DR or N/A.
                GPS - location is received from GPS
                ODO - location is calculated based on odometer value
                MAN - location is specified manually
                DR - location is calculated using dead reckoning (used in tunnels and other locations without GPS signal)
                N/A - location is unavailable
stop    str     ID of the stop related to the event (e.g. ID of the stop where the vehicle departed from in case of dep event or the stop
                 where the vehicle currently is in case of vp event). null if the event is not related to any stop.
route   str     ID of the route the vehicle is currently running on. Matches route_id in the topic.
occu    int     Integer describing passenger occupancy level of the vehicle. Valid values are on interval [0, 100].
                Currently passenger occupancy level is only available for Suomenlinna ferries as a proof-of-concept.
                The value will be available shortly after departure when the ferry operator has registered passenger count for the journey.


"""

import argparse
import gzip
import json
import logging
import os
import pathlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import sentry_sdk
from tqdm import tqdm


def sentry_init(args: argparse.Namespace):
    """Initialize Sentry SDK if DSN is provided."""
    dsn = args.sentry_dsn or os.environ.get("SENTRY_DSN")
    if dsn and dsn.startswith("https"):
        try:
            sentry_sdk.init(
                dsn=dsn,
                # Set traces_sample_rate to 1.0 to capture 100%
                # of transactions for performance monitoring.
                traces_sample_rate=1.0,
                # Set profiles_sample_rate to 1.0 to profile 100%
                # of sampled transactions.
                # We recommend adjusting this value in production.
                profiles_sample_rate=1.0,
            )
            logging.info("Sentry SDK initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize Sentry SDK: {e}")
    else:
        logging.info("Sentry DSN not found or invalid, skipping Sentry initialization.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert HSL HFP MQTT data dump files to GeoParquet format.")
    parser.add_argument(
        "--rt-files", nargs="+", required=True, help="List of HFP data files (text or gzip) or directory."
    )
    # parser.add_argument(
    #     "--bbox",
    #     default="25.007142,60.221397,25.023972,60.229863",
    #     help="Bounding box for filtering (lon_min,lat_min,lon_max,lat_max).",
    # )
    # parser.add_argument("--dir", type=str, help='Direction to filter (e.g., "1" or "2").')
    parser.add_argument(
        "--outfile",
        nargs="+",
        choices=["geoparquet", "parquet", "csv"],
        default=["geoparquet"],
        help="Output file format (geoparquet, parquet or csv)",
    )
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument(
        "--sentry-dsn", help="Sentry DSN for error tracking. Can be set as environment variable SENTRY_DSN."
    )
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


def process_file(file_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Process a single HFP data file, separating VP records and other lines.

    Args:
        file_path: Path to the HFP data file

    Returns:
        A tuple containing:
        - List of parsed VP records
        - List of raw lines for non-VP messages
    """
    vp_records = []
    other_lines = []

    # Check if the file is gzipped
    open_func, mode = (gzip.open, "rt") if file_path.endswith(".gz") else (open, "r")

    # Process the file with a progress bar
    with open_func(file_path, mode) as f:
        for line in tqdm(f, desc=f"Processing {Path(file_path).name}", unit="lines", total=None):
            record = parse_hfp_line(line.strip())  # Ensure no leading/trailing whitespace
            if record is None:
                # If parsing failed or it's not a VP message, store the raw line
                # We check if '/vp/' is in the topic part just in case parsing failed for a VP line
                json_start = line.find("{")
                topic = line[:json_start].strip() if json_start != -1 else ""
                if "/vp/" in topic:
                    logging.warning(f"Could not parse potential VP line, storing raw: {line.strip()}")
                other_lines.append(line.strip())  # Store original line
            else:
                # Successfully parsed VP record
                vp_records.append(record)

    return vp_records, other_lines


def process_files(file_paths: List[str], chunk_size: int = 100000) -> Tuple[pd.DataFrame, List[str]]:
    """
    Process multiple HFP files in chunks, separating VP data and other lines.

    Args:
        file_paths: List of HFP data files
        chunk_size: Number of records to process before creating a DataFrame for VP data

    Returns:
        A tuple containing:
        - DataFrame with all the processed VP records
        - List of all raw lines for non-VP messages
    """
    all_dfs = []
    all_other_lines = []
    current_vp_chunk = []

    for file_path in file_paths:
        logging.info(f"Processing file: {file_path}")

        # Process each file
        vp_records, other_lines_from_file = process_file(file_path)

        # Add records/lines to respective lists
        current_vp_chunk.extend(vp_records)
        all_other_lines.extend(other_lines_from_file)

        # If the VP chunk is large enough, convert to DataFrame and add to the list
        if len(current_vp_chunk) >= chunk_size:
            df_chunk = pd.DataFrame(current_vp_chunk)
            all_dfs.append(df_chunk)
            current_vp_chunk = []  # Reset the chunk

    # Process the remaining VP records
    if current_vp_chunk:
        df_chunk = pd.DataFrame(current_vp_chunk)
        all_dfs.append(df_chunk)

    # Combine all VP DataFrames
    if not all_dfs:
        vp_df = pd.DataFrame()  # Return empty DataFrame if no VP data
    else:
        vp_df = pd.concat(all_dfs, ignore_index=True)
        logging.info(f"Combined VP DataFrame head:\n{vp_df.head()}")

    logging.info(f"Total non-VP lines collected: {len(all_other_lines)}")
    return vp_df, all_other_lines


def save_dataframe(df: pd.DataFrame, outfile: Path, outfile_format: List[str]):
    """Save the DataFrame to the specified output file(s)."""

    # Ensure the output directory exists
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # Save based on file extension
    for of in outfile_format:
        full_path = str(outfile) + "." + of
        print(f"Saving data to {full_path}")
        if of.endswith("geoparquet"):
            # Create geometry column from lat/long using geopandas helper function
            # This correctly handles missing values (pd.NA)
            geometry = gpd.points_from_xy(df["long"], df["lat"])
            # Convert to GeoDataFrame, in WGS84 coordinate system
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            # Drop longitude and latitude columns now that geometry is created
            gdf = gdf.drop(columns=["long", "lat"])
            # Save as GeoParquet
            gdf.to_parquet(full_path, index=False)
        elif of.endswith("parquet"):
            df.to_parquet(full_path)
        elif of.endswith("csv"):
            df.to_csv(full_path, index=False)
        else:
            raise ValueError("Output file must have .parquet, .geoparquet, or .csv extension")

    print(f"Data saved successfully. Total records: {len(df)}")


def save_other_messages(lines: List[str], outfile_path: Path):
    """Save the list of non-VP message lines to a gzipped text file."""
    # Ensure the output directory exists
    outfile_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(lines)} other messages to {outfile_path}")
    try:
        with gzip.open(outfile_path, "wt", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        print(f"Other messages saved successfully to {outfile_path}")
    except Exception as e:
        logging.error(f"Failed to save other messages to {outfile_path}: {e}")


def parse_route_from_filename(filename: str) -> Optional[str]:
    """
    Parse route number from filename.

    Args:
        filename: Input filename (e.g. '1234-2025-03-14T13.txt.gz' or '31M1B-2025-03-14T13.txt.gz')

    Returns:
        First 4 characters of route or None if parsing fails
    """
    try:
        # Split by dash and take the first part
        route_str = filename.split("-")[0]
        # Take first 4 characters of the route
        route_base = route_str[:4]
        return route_base if route_base else None
    except (IndexError, ValueError):
        return None


def get_route_group(route: str) -> str:
    """
    Determine the route group for aggregation.

    Args:
        route: Route identifier (first 4 characters)

    Returns:
        Route group identifier (either specific route or group range)
    """
    try:
        # Try to convert the numeric part to integer for grouping
        if "1001" <= route <= "4999":
            return route  # Return original route string for individual files
        elif "5000" <= route <= "9999":
            return route[:1] + "xxx"  # Replace last 3 digits with xxx
        else:
            return "other"
    except ValueError:
        return "other"  # For routes that can't be converted to numbers


def group_input_files(file_paths: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    Group input files by date and route/route group.

    Args:
        file_paths: List of input file paths or directory

    Returns:
        Nested dictionary: {date: {route_group: [files]}}
    """
    grouped_files: Dict[str, Dict[str, List[str]]] = {}

    # Process all file paths and expand directories
    expanded_paths = []
    for path in file_paths:
        if pathlib.Path(path).is_dir():
            # Add all files from directory
            expanded_paths.extend([str(f) for f in Path(path).iterdir()])
        else:
            expanded_paths.append(path)
    file_paths = expanded_paths

    for file_path in file_paths:
        filename = Path(file_path).name

        # Parse route number
        route = parse_route_from_filename(filename)
        if route is None:
            logging.warning(f"Could not parse route from filename: {filename}")
            continue

        # Parse date from filename (e.g. '1234-2025-03-14T13.txt.gz')
        try:
            date_str = "-".join(filename.split("-")[1:]).split("T")[0]
        except IndexError:
            logging.warning(f"Could not parse date from filename: {filename}")
            continue

        route_group = get_route_group(route)

        # Initialize nested dictionaries if they don't exist
        if date_str not in grouped_files:
            grouped_files[date_str] = {}
        if route_group not in grouped_files[date_str]:
            grouped_files[date_str][route_group] = []

        grouped_files[date_str][route_group].append(file_path)
    # Sort route groups within each date
    for date in grouped_files:
        grouped_files[date] = dict(sorted(grouped_files[date].items()))
    return grouped_files


def process_route_group(date: str, route_group: str, files: List[str], output_dir: str, output_formats: List[str]):
    """
    Process all files for a single route group, saving VP data and other messages separately.

    Args:
        date: Date string (YYYY-MM-DD)
        route_group: Route group identifier
        files: List of input files
        output_dir: Base output directory
        output_formats: List of output formats to generate for VP data
    """
    print(f"Processing route group: {route_group} for date {date}")
    # Process all files and create the DataFrame for VP data and list for other lines
    df_vp, other_lines = process_files(files)

    # Define base output path components
    date_nodash = date.replace("-", "")
    output_base_path = pathlib.Path(output_dir) / date / f"{route_group}_{date_nodash}"

    # Process and save VP data if any exists
    if not df_vp.empty:
        print(f"Processing {len(df_vp)} VP records for {route_group} on {date}")
        # Convert timestamp and optimize data types
        # Store original tst for logging purposes before converting to datetime
        original_tst = df_vp["tst"].copy()
        # Convert tst to datetime, coercing errors to NaT and handling mixed ISO8601 formats
        df_vp["ts"] = pd.to_datetime(df_vp["tst"], format="ISO8601", errors="coerce")

        # Identify rows with NaT timestamps
        invalid_ts_mask = df_vp["ts"].isna()
        if invalid_ts_mask.any():
            invalid_rows = df_vp[invalid_ts_mask]
            # Log the number of invalid records found
            logging.warning(
                f"Found {len(invalid_rows)} records with invalid timestamps in {files} for {route_group} on {date}. Removing them."
            )
            for index, row in invalid_rows.iterrows():
                # Use the original tst value for logging
                original_invalid_tst = original_tst.loc[index]
                logging.warning(
                    f"Invalid timestamp format found and removed: '{original_invalid_tst}' "
                    f"(route_group: {route_group}, date: {date}, original_index: {index})"
                )
            # Remove rows with NaT timestamps
            df_vp = df_vp[~invalid_ts_mask].copy()  # Use copy to avoid SettingWithCopyWarning

        # Drop rows with missing latitude or longitude BEFORE further processing
        original_count = len(df_vp)
        df_vp = df_vp.dropna(subset=["lat", "long"])
        dropped_count = original_count - len(df_vp)
        if dropped_count > 0:
            logging.info(f"Dropped {dropped_count} rows with missing lat/long values for {route_group} on {date}.")

        # Handle potential errors during datetime conversion if needed
        # df_vp["oday_start"] = pd.to_datetime(df_vp["oday"] + " " + df_vp["start"], errors='coerce') # Example with error handling
        df_vp["oday_start"] = pd.to_datetime(df_vp["oday"] + " " + df_vp["start"])

        # Drop original timestamp fields
        # Keep original tst column until after error handling for logging
        df_vp = df_vp.drop(columns=["tst", "tsi", "oday", "start"])

        # Fill NA values with sensible defaults before type conversion
        na_fill_values = {
            "dir": 0,  # dir can be 1 or 2
            "drst": False,  # Door status, False means closed
            "hdg": -1,  # Heading, -1 means unknown
            "occu": 0,  # Occupancy, 0-100
            "oper": 0,  # Operator, 0 means unknown
            "veh": 0,  # Vehicle, 0 means unknown
            "lat": pd.NA,  # Use pandas NA for float nulls
            "long": pd.NA,  # Use pandas NA for float nulls
            "acc": pd.NA,
            "dl": pd.NA,
            "odo": pd.NA,
            "stop": pd.NA,
            "jrn": pd.NA,
            "line": pd.NA,
            "spd": pd.NA,
        }
        # df_vp = df_vp.fillna(na_fill_values) # Careful: fillna converts int columns with NA to float
        for col, val in na_fill_values.items():
            if col in df_vp.columns:
                # Skip drst, handle it specifically later
                if col == "drst":
                    continue
                df_vp[col] = df_vp[col].fillna(val)

        # Handle drst column explicitly to convert 0/1 or 0.0/1.0 to boolean
        if "drst" in df_vp.columns:
            # First convert to numeric, coercing errors to NaN (becomes pd.NA)
            df_vp["drst"] = pd.to_numeric(df_vp["drst"], errors="coerce")
            # Map numeric values to boolean, keep NA as NA
            # 1.0 means True (doors open), 0.0 means False (doors closed)
            df_vp["drst"] = (df_vp["drst"] == 1.0).astype("boolean")

        # Check dl column for values outside Int16 range before conversion
        if "dl" in df_vp.columns:
            # Ensure dl is numeric first, coercing errors
            df_vp["dl"] = pd.to_numeric(df_vp["dl"], errors="coerce")
            # Define Int16 limits
            int16_min = -32768
            int16_max = 32767
            # Set values outside the range to pd.NA
            df_vp.loc[~df_vp["dl"].between(int16_min, int16_max, inclusive="both"), "dl"] = pd.NA

        # Convert data types
        dtype_conversions = {
            "dir": "UInt8",  # Use nullable Int type
            # "drst": "boolean", # Already handled above
            "hdg": "Int16",  # Use nullable Int type
            "occu": "UInt8",  # Use nullable Int type
            "oper": "UInt16",  # Use nullable Int type
            "veh": "UInt16",  # Use nullable Int type
            "dl": "Int16",  # Nullable signed integer for delay (-32k to +32k seconds)
            "odo": "UInt32",  # Nullable unsigned integer for odometer
            "stop": "UInt32",  # Nullable unsigned integer for stop ID
            "jrn": "UInt16",  # Nullable unsigned integer for journey ID
            "line": "UInt16",  # Nullable unsigned integer for line ID
            "lat": "float64",  # Latitude (will use nullable Float64)
            "long": "float64",  # Longitude (will use nullable Float64)
            "spd": "float32",  # Speed in m/s (will use nullable Float32)
            "acc": "float32",  # Acceleration in m/s^2 (will use nullable Float32)
        }
        for col, dtype in dtype_conversions.items():
            if col in df_vp.columns:
                try:
                    # Use pd.to_numeric for float types to handle errors and NA correctly
                    if dtype.startswith("float"):
                        # Use Float64 for coordinates, Float32 for others
                        pd_dtype = "Float64" if col in ["lat", "long"] else "Float32"
                        df_vp[col] = pd.to_numeric(df_vp[col], errors="coerce").astype(pd_dtype)
                    else:
                        df_vp[col] = df_vp[col].astype(dtype)  # Existing nullable Int types handle NA
                except Exception as e:
                    logging.warning(f"Could not convert column {col} to {dtype}: {e}")

        # Convert categorical columns
        categorical_columns = ["loc", "desi", "route"]
        for col in categorical_columns:
            if col in df_vp.columns:
                df_vp[col] = df_vp[col].astype("category")

        # Sort by timestamp and vehicle
        # df_vp = df_vp.sort_values(["ts", "veh"]) # Sorting can be slow, consider if necessary here or later
        df_vp = df_vp.sort_values(["ts"])  # Sort primarily by time

        # Save the processed VP DataFrame
        save_dataframe(df_vp, output_base_path, output_formats)
    else:
        print(f"No VP records found for {route_group} on {date}. Skipping VP file save.")

    # Save other messages if any exist
    if other_lines:
        output_path_other = output_base_path.parent / f"{output_base_path.name}_other_messages.txt.gz"
        save_other_messages(other_lines, output_path_other)
    else:
        print(f"No other messages found for {route_group} on {date}. Skipping other messages file save.")


def main():
    args = parse_args()

    # Initialize Sentry if configured
    sentry_init(args)

    # Group files by date and route/route group
    grouped_files = group_input_files(args.rt_files)

    # Process each date and route group
    for date in grouped_files:
        print(f"\nDate: {date}")
        print("Route groups:", sorted(list(grouped_files[date].keys())))

        for route_group in grouped_files[date]:
            process_route_group(
                date=date,
                route_group=route_group,
                files=grouped_files[date][route_group],
                output_dir=args.outdir,
                output_formats=args.outfile,
            )


if __name__ == "__main__":
    main()
