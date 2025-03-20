"""
This script is used to process HFP data from the HSL HFP MQTT data dump files to GeoParquet format.

Sample lines in raw data files:

/hfp/v2/journey/ongoing/vp/tram/0040/00436/1003/1/Pikku Huopalahti/12:50/1160403/4/60;24/28/09/38 {"VP": {"desi": "3", "dir": "1", "oper": 40, "veh": 436, "tst": "2025-03-12T11:24:51.251Z", "tsi": 1741778691, "spd": 5.5, "hdg": 268, "lat": 60.203263, "long": 24.898389, "acc": 0.11, "dl": 262, "odo": 9070, "drst": 0, "oday": "2025-03-12", "jrn": 762, "line": 31, "start": "12:50", "loc": "GPS", "stop": null, "route": "1003", "occu": 0}}
/hfp/v2/journey/ongoing/vp/bus/0022/01400/2212/1/Kauniala/11:26/2252204/5/60;24/27/08/15 {"VP": {"desi": "212", "dir": "1", "oper": 22, "veh": 1400, "tst": "2025-03-12T10:00:04.751Z", "tsi": 1741773604, "spd": 1.08, "hdg": 268, "lat": 60.201181, "long": 24.785763, "acc": 1.08, "dl": -120, "odo": 12579, "drst": 0, "oday": "2025-03-12", "jrn": 525, "line": 244, "start": "11:26", "loc": "GPS", "stop": 2252204, "route": "2212", "occu": 0}}
/hfp/v2/journey/ongoing/vp/bus/0018/01095/1065/1/Veräjälaakso/11:55/1111117/5/60;24/19/75/71 {"VP": {"desi": "65", "dir": "1", "oper": 6, "veh": 1095, "tst": "2025-03-12T10:00:02.751Z", "tsi": 1741773602, "spd": 7.4, "hdg": 354, "lat": 60.177785, "long": 24.951814, "acc": -0.38, "dl": -70, "odo": 1139, "drst": 0, "oday": "2025-03-12", "jrn": 1089, "line": 850, "start": "11:55", "loc": "GPS", "stop": null, "route": "1065", "occu": 0}}

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
import pathlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm


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


def process_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Process a single HFP data file with filtering.

    Args:
        file_path: Path to the HFP data file

    Returns:
        List of parsed records
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

            results.append(record)

    return results


def process_files(file_paths: List[str], chunk_size: int = 100000) -> pd.DataFrame:
    """
    Process multiple HFP files in chunks to avoid loading everything into memory.

    Args:
        file_paths: List of HFP data files
        chunk_size: Number of records to process before creating a DataFrame

    Returns:
        DataFrame with all the processed records
    """
    all_dfs = []
    current_chunk = []

    for file_path in file_paths:
        print(f"Processing file: {file_path}")

        # Process each file
        records = process_file(file_path)

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


def save_dataframe(df: pd.DataFrame, outfile: Path, outfile_format: List[str]):
    """Save the DataFrame to the specified output file(s)."""

    # Ensure the output directory exists
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # Save based on file extension
    for of in outfile_format:
        full_path = str(outfile) + "." + of
        print(f"Saving data to {full_path}")
        if of.endswith("geoparquet"):
            # Create geometry column from lat/long
            geometry = [Point(xy) for xy in zip(df["long"], df["lat"])]
            # Convert to GeoDataFrame, in WGS84 coordinate system
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            # Save as GeoParquet
            gdf.to_parquet(full_path, index=False)
        elif of.endswith("parquet"):
            df.to_parquet(full_path)
        elif of.endswith("csv"):
            df.to_csv(full_path, index=False)
        else:
            raise ValueError("Output file must have .parquet, .geoparquet, or .csv extension")

    print(f"Data saved successfully. Total records: {len(df)}")


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
    Process all files for a single route group.

    Args:
        date: Date string (YYYY-MM-DD)
        route_group: Route group identifier
        files: List of input files
        output_dir: Base output directory
        output_formats: List of output formats to generate
    """
    print(f"Processing route group: {route_group}")
    # Process all files and create the DataFrame
    df = process_files(files)

    if not df.empty:
        # Convert timestamp and optimize data types
        df["ts"] = pd.to_datetime(df["tst"])
        df["oday_start"] = pd.to_datetime(df["oday"] + " " + df["start"])

        # Drop original timestamp fields
        df = df.drop(columns=["tst", "tsi", "oday", "start"])

        # Fill NA values with sensible defaults before type conversion
        na_fill_values = {
            "dir": 1,  # Oletussuunta 1
            "drst": False,  # Oletuksena ovet kiinni
            "hdg": -1,  # -1 tarkoittaa puuttuvaa suuntaa
            "occu": 0,  # Oletuksena tyhjä
            "oper": 0,  # Tuntematon operaattori
            "veh": 0,  # Tuntematon ajoneuvo
        }
        df = df.fillna(na_fill_values)

        # Convert data types
        dtype_conversions = {
            "dir": "uint8",
            "drst": "bool",
            "hdg": "int16",
            "occu": "uint8",
            "oper": "uint16",
            "veh": "uint16",
        }
        df = df.astype(dtype_conversions)

        # Convert categorical columns
        categorical_columns = ["loc", "desi", "route"]
        for col in categorical_columns:
            df[col] = df[col].astype("category")

        # Sort by timestamp and vehicle
        df = df.sort_values(["ts", "veh"])

    # Create output path and save
    date_str = date.replace("-", "")
    output_path = pathlib.Path(output_dir) / date / f"{route_group}_{date_str}"
    save_dataframe(df, output_path, output_formats)


def main():
    args = parse_args()

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
