"""
Script for visualizing HSL vehicle GPS tracks on a map using Folium.

This script takes cleaned GPS track data (from track_data_cleaner.py) and creates
an interactive Folium map. It can also apply additional filters at visualization
time if needed.

The script creates an HTML map with:
- Vehicle tracks as colored lines
- Interactive legend to show/hide individual vehicles
- Tooltips with trip information
- Optional geometry boundaries from GeoJSON configuration
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Union

import folium
import geopandas as gpd
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


def generate_color_from_veh(veh_number: int) -> str:
    """Generate a consistent color for a vehicle number using hash."""
    # Convert to string and hash
    hash_object = hashlib.md5(str(veh_number).encode())
    hex_hash = hash_object.hexdigest()

    # Take first 6 characters as RGB hex color
    color = f"#{hex_hash[:6]}"
    return color


def create_vehicle_tracks(df: gpd.GeoDataFrame, time_threshold_seconds: int = 5, min_points: int = 10) -> dict:
    """Create LineString tracks for each vehicle from GPS points, splitting trips based on time gaps.

    Args:
        df: GeoDataFrame with vehicle GPS data
        time_threshold_seconds: Maximum time gap in seconds between consecutive points
                               before splitting into separate trips (default: 5)

    Returns:
        Dictionary with vehicle numbers as keys and lists of trip dictionaries as values.
        Each trip contains linestring, timestamps, color, and metadata.
    """
    tracks = {}

    for veh in df["veh"].unique():
        veh_data = df[df["veh"] == veh].copy()

        # Sort by timestamp to ensure correct order
        veh_data = veh_data.sort_values("ts")

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
                if len(current_trip_indices) >= min_points:  # Need at least `min_points` points for a line
                    trips.append(current_trip_indices)
                else:
                    logging.debug(f"Skipping trip with less than {min_points} points")
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
                    "color": generate_color_from_veh(veh),
                    "trip_id": trip_idx,
                    "point_count": len(coords),
                    "vehicle_id": veh,
                }
                vehicle_trips.append(trip_info)

        if vehicle_trips:  # Only add if we have valid trips
            tracks[veh] = vehicle_trips

    return tracks


def filter_dataframe(df: gpd.GeoDataFrame, args: argparse.Namespace) -> gpd.GeoDataFrame:
    """Apply visualization-time filters to the dataframe.

    Args:
        df: Input GeoDataFrame
        args: Command line arguments

    Returns:
        Filtered GeoDataFrame
    """
    original_size = len(df)
    logging.info(f"Starting with {original_size} GPS points")

    # Apply bounding box filter
    if args.bbox:
        bbox = [float(coord) for coord in args.bbox.split(",")]
        df = df[df.geometry.within(box(bbox[0], bbox[1], bbox[2], bbox[3]))]
        logging.info(f"After bounding box filter: {len(df)} points ({len(df) / original_size * 100:.1f}%)")

    # Apply vehicle filter
    if args.veh:
        df = df[df["veh"].isin(args.veh)]
        logging.info(f"After vehicle filter: {len(df)} points ({len(df) / original_size * 100:.1f}%)")

    # Apply direction filter
    if args.dir:
        df = df[df["dir"] == args.dir]
        logging.info(f"After direction filter: {len(df)} points ({len(df) / original_size * 100:.1f}%)")

    return df


def create_folium_map(
    tracks: dict,
    output_path: str,
    geometry_to_display: Optional[Union[LineString, Polygon]] = None,
    geometry_name: str = "Geometry",
) -> None:
    """Create a Folium map with vehicle tracks, tooltips and interactive legend.

    Args:
        tracks: Dictionary of vehicle tracks
        output_path: Path to save the HTML file
        geometry_to_display: Optional geometry to display on map (e.g., filter boundaries)
        geometry_name: Name for the geometry in the legend
    """
    if not tracks:
        logging.warning("No tracks to visualize")
        return

    # Calculate map center from all track coordinates
    all_coords = []
    for vehicle_trips in tracks.values():
        for trip in vehicle_trips:
            coords = list(trip["linestring"].coords)
            all_coords.extend(coords)

    if all_coords:
        center_lon = sum(coord[0] for coord in all_coords) / len(all_coords)
        center_lat = sum(coord[1] for coord in all_coords) / len(all_coords)
    else:
        # Default to Helsinki center
        center_lon, center_lat = 24.9384, 60.1699

    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB Positron")

    # Add geometry to map if provided
    if geometry_to_display is not None:
        if isinstance(geometry_to_display, Polygon):
            # Convert polygon coordinates for Folium (lat, lon format)
            coords = [(coord[1], coord[0]) for coord in geometry_to_display.exterior.coords]

            folium.Polygon(
                locations=coords,
                color="red",
                weight=2,
                opacity=0.8,
                fill=True,
                fillColor="red",
                fillOpacity=0.1,
                tooltip=f"{geometry_name} boundary",
            ).add_to(m)

        elif isinstance(geometry_to_display, LineString):
            # Convert linestring coordinates for Folium (lat, lon format)
            coords = [(coord[1], coord[0]) for coord in geometry_to_display.coords]

            folium.PolyLine(
                locations=coords,
                color="red",
                weight=3,
                opacity=0.8,
                tooltip=f"{geometry_name} line",
            ).add_to(m)

    # Add tracks to map with unique IDs for JavaScript control
    for veh, vehicle_trips in tracks.items():
        for trip in vehicle_trips:
            coords = [(coord[1], coord[0]) for coord in trip["linestring"].coords]  # Folium expects (lat, lon)

            # Format timestamps for tooltip
            first_time_str = trip["first_time"].strftime("%Y-%m-%d %H:%M:%S")
            last_time_str = trip["last_time"].strftime("%Y-%m-%d %H:%M:%S")

            # Create tooltip text
            tooltip_text = f"""
            Vehicle: {veh}<br>
            Trip: {trip["trip_id"] + 1}<br>
            Points: {trip["point_count"]}<br>
            First timestamp: {first_time_str}<br>
            Last timestamp: {last_time_str}
            """

            # Add line to map with unique class for JavaScript control
            line = folium.PolyLine(
                locations=coords,
                color=trip["color"],
                weight=3,
                opacity=0.8,
                tooltip=tooltip_text,
                className=f"vehicle-{veh}",
            )
            line.add_to(m)

    # Create interactive legend HTML with JavaScript
    legend_html = """
    <div id="legend" style="position: fixed;
                top: 10px; right: 10px; width: 220px; height: auto;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px; border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.5);">
    <h4 style="margin-top:0;">Vehicle Legend <span id="toggle-all" style="font-size:12px; cursor:pointer; color:blue;">[Show All]</span></h4>
    """

    # Add clickable legend items
    for veh in sorted(tracks.keys()):
        # Get color from first trip (all trips for same vehicle have same color)
        color = tracks[veh][0]["color"]
        trip_count = len(tracks[veh])
        legend_html += f"""
        <p style="cursor: pointer; user-select: none; margin-bottom: 0px;" onclick="toggleVehicle({veh})" id="legend-{veh}">
            <span style="color:{color}; font-size:16px;">&#9632;</span> Vehicle {veh} ({trip_count} trips)
        </p>
        """

    legend_html += (
        """
    </div>

    <script>
    // Track which vehicles are currently visible
    var vehicleStates = {};
    var allVisible = true;

    // Initialize all vehicles as visible
    """
        + "\n".join([f"vehicleStates[{veh}] = true;" for veh in tracks.keys()])
        + """

    function toggleVehicle(veh) {
        // Get all polylines with the vehicle class
        var elements = document.getElementsByClassName('vehicle-' + veh);
        var isVisible = vehicleStates[veh];

        // Toggle visibility
        for (var i = 0; i < elements.length; i++) {
            if (isVisible) {
                elements[i].style.display = 'none';
            } else {
                elements[i].style.display = 'block';
            }
        }

        // Update state
        vehicleStates[veh] = !isVisible;

        // Update legend item appearance
        var legendItem = document.getElementById('legend-' + veh);
        if (!isVisible) {
            legendItem.style.opacity = '1.0';
            legendItem.style.fontWeight = 'bold';
        } else {
            legendItem.style.opacity = '0.5';
            legendItem.style.fontWeight = 'normal';
        }

        // Check if all are visible or hidden
        updateToggleAllButton();
    }

    function toggleAll() {
        var anyVisible = Object.values(vehicleStates).some(v => v);

        if (anyVisible) {
            // Hide all
            """
        + "\n".join([f"toggleVehicleIfVisible({veh});" for veh in tracks.keys()])
        + """
        } else {
            // Show all
            """
        + "\n".join([f"toggleVehicleIfHidden({veh});" for veh in tracks.keys()])
        + """
        }
        updateToggleAllButton();
    }

    function toggleVehicleIfVisible(veh) {
        if (vehicleStates[veh]) {
            toggleVehicle(veh);
        }
    }

    function toggleVehicleIfHidden(veh) {
        if (!vehicleStates[veh]) {
            toggleVehicle(veh);
        }
    }

    function updateToggleAllButton() {
        var allVisible = Object.values(vehicleStates).every(v => v);
        var anyVisible = Object.values(vehicleStates).some(v => v);
        var toggleButton = document.getElementById('toggle-all');

        if (allVisible) {
            toggleButton.textContent = '[Hide All]';
        } else if (anyVisible) {
            toggleButton.textContent = '[Show All]';
        } else {
            toggleButton.textContent = '[Show All]';
        }
    }

    // Add click handler for toggle all button
    document.getElementById('toggle-all').onclick = toggleAll;

    // Initial state
    updateToggleAllButton();
    </script>
    """
    )

    # Add legend to map
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save map
    m.save(output_path)
    logging.info(f"Interactive map saved to {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize HSL vehicle GPS tracks on an interactive map")

    # Input/Output
    parser.add_argument("--input", type=str, required=True, help="Path to the cleaned GPS data parquet file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the HTML map file")

    # Visualization-time filters (optional additional filtering)
    parser.add_argument("--bbox", type=str, help="Additional bounding box filter 'lon_min,lat_min,lon_max,lat_max'")
    parser.add_argument("--veh", type=int, nargs="+", help="Filter for specific vehicle number(s)")
    parser.add_argument("--dir", type=int, help="Filter for specific direction number")

    # Track processing (can override cleaned data settings)
    parser.add_argument(
        "--time-threshold",
        type=int,
        default=5,
        help="Time threshold in seconds for splitting vehicle trips (default: 5)",
    )
    # Track filtering
    parser.add_argument(
        "--min-points",
        type=int,
        default=10,
        help="Minimum number of points in track segment (default: 10)",
    )

    # Geometry display
    parser.add_argument(
        "--show-geometry", type=str, help="Name of geometry feature to display on map (from GeoJSON config)"
    )
    parser.add_argument(
        "--geometry-config",
        type=str,
        default="track_geometries.geojson",
        help="Path to GeoJSON geometry configuration file (default: track_geometries.geojson)",
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

    return args


def main():
    """Main visualization function."""
    args = parse_args()

    # Read cleaned GPS data
    logging.info(f"Reading GPS data from {args.input}")
    df = gpd.read_parquet(args.input)

    if df.empty:
        logging.error("Input data is empty")
        return

    # Apply additional filters if specified
    if args.bbox or args.veh or args.dir:
        logging.info("Applying additional visualization filters")
        df = filter_dataframe(df, args)

        if df.empty:
            logging.warning("No data remaining after visualization filters")
            return

    # Create vehicle tracks
    logging.info(f"Creating tracks for {len(df['veh'].unique())} vehicles with {args.time_threshold}s time threshold")
    tracks = create_vehicle_tracks(df, time_threshold_seconds=args.time_threshold, min_points=args.min_points)

    if not tracks:
        logging.warning("No tracks were created. Check your data and filters.")
        return

    # Load geometry for display if specified
    geometry_to_display = None
    geometry_name = "Geometry"
    if args.show_geometry:
        try:
            logging.info(f"Loading geometry '{args.show_geometry}' for display")
            geometry, properties = load_geometry_feature(args.geometry_config, args.show_geometry)
            geometry_to_display = geometry
            geometry_name = properties.get("display_name", args.show_geometry)
            logging.info(f"Will display {type(geometry).__name__} geometry: {geometry_name}")
        except (FileNotFoundError, KeyError, ValueError) as e:
            logging.warning(f"Could not load geometry for display: {e}")

    # Create and save the map
    total_trips = sum(len(vehicle_trips) for vehicle_trips in tracks.values())
    create_folium_map(tracks, args.output, geometry_to_display, geometry_name)

    logging.info("Visualization complete:")
    logging.info(f"  - {len(tracks)} vehicles visualized")
    logging.info(f"  - {total_trips} track segments")
    logging.info(f"  - Map saved to: {args.output}")


if __name__ == "__main__":
    main()
