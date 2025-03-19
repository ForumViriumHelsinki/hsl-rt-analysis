"""
Script for visualizing HSL MQTT data on a map using Folium.
Supports point, line and heatmap visualization types.
Used for debugging data dump files.

Usage:
python mqtt_datadump_check.py --log DEBUG --files data/raw/all/2025-03-12/1055* --limit 1000000 --output bus_55_route.html --type lines
"""

import argparse
import gzip
import json
import logging
from typing import List

import branca
import folium
from folium import plugins


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize HSL MQTT data on a map")
    parser.add_argument(
        "--log", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Logging level"
    )
    parser.add_argument("--limit", type=int, default=10000, help="Maximum number of points to plot")
    parser.add_argument("--files", nargs="+", required=True, help="Input MQTT dump files")
    parser.add_argument("--output", type=str, default="mqtt_visualization.html", help="Output HTML file name")
    parser.add_argument(
        "--type", nargs="+", choices=["points", "heatmap", "lines"], default=["points"], help="Visualization type(s)"
    )
    args = parser.parse_args()
    # Set logging level based on argument
    logging.basicConfig(
        level=getattr(logging, args.log),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return args


def parse_mqtt_line(line: str) -> dict | None:
    """Parse MQTT line and extract vehicle data.

    Returns:
        dict: Dictionary containing lat, lon, acc, spd or None if parsing fails
    """
    try:
        json_start = line.find("{")
        if json_start == -1:
            return None

        topic = line[:json_start].strip()
        payload = line[json_start:].strip()

        # Only process VP messages
        if "vp" not in topic.lower():
            return None

        # Parse JSON payload
        data = json.loads(payload)

        # Extract coordinates
        if "VP" in data:
            vp_data = data["VP"]
            lat = vp_data.get("lat")
            lon = vp_data.get("long")

            if lat is None or lon is None:
                return None

            return {
                "lat": lat,
                "lon": lon,
                "acc": vp_data.get("acc", None),
                "spd": vp_data.get("spd", None),
                "veh": vp_data.get("veh", None),
                "tsi": vp_data.get("tsi", None),
            }

    except Exception as e:
        logger.error(f"Error parsing line: {e}")
    return None


def collect_vehicle_data(files: List[str], limit: int) -> List[dict]:
    """Collect vehicle data from MQTT dump files.

    Args:
        files: List of input file paths
        limit: Maximum number of points to process

    Returns:
        List of dictionaries containing vehicle data
    """
    vehicle_data: List[dict] = []

    for file_path in files:
        logger.info(f"Processing file: {file_path}")
        open_func = gzip.open if file_path.endswith(".gz") else open

        with open_func(file_path, "rt") as f:
            for line in f:
                if len(vehicle_data) >= limit:
                    logging.warning(f"Reached limit of {limit} coordinates, stopping")
                    break

                data = parse_mqtt_line(line)
                if data:
                    vehicle_data.append(data)

    return vehicle_data


def create_map(vehicle_data: List[dict], viz_type: str = "points") -> folium.Map:
    """Create a Folium map with the given vehicle data."""
    center_lat = sum(data["lat"] for data in vehicle_data) / len(vehicle_data)
    center_lon = sum(data["lon"] for data in vehicle_data) / len(vehicle_data)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")

    if "heatmap" in viz_type:
        # Prepare data for heatmap
        heat_data = [[data["lat"], data["lon"]] for data in vehicle_data]
        plugins.HeatMap(heat_data).add_to(m)
    if "lines" in viz_type:
        # Organize data by vehicle and split routes into segments
        vehicle_routes: dict[str, list[list[list[float]]]] = {}
        for data in vehicle_data:
            veh = data.get("veh", "unknown")
            if veh not in vehicle_routes:
                vehicle_routes[veh] = [[]]  # List of route segments

            current_route = vehicle_routes[veh][-1]  # Last route segment

            # If route segment has points, check time difference
            if current_route and len(current_route) > 0:
                last_point = current_route[-1]
                last_tsi = last_point[2]  # tsi is stored as third value
                current_tsi = data.get("tsi", 0)

                # If time difference is more than 5 seconds, start new route segment
                if abs(current_tsi - last_tsi) > 5:
                    vehicle_routes[veh].append([])
                    current_route = vehicle_routes[veh][-1]

            # Add point to current route segment [lat, lon, tsi]
            current_route.append([data["lat"], data["lon"], data.get("tsi", 0)])

        # Create random color for each vehicle using hash function
        def get_color_from_string(s: str) -> str:
            import hashlib

            hash_value = int(hashlib.md5(str(s).encode()).hexdigest(), 16)
            r = (hash_value & 0xFF0000) >> 16
            g = (hash_value & 0x00FF00) >> 8
            b = hash_value & 0x0000FF
            return f"#{r:02x}{g:02x}{b:02x}"

        # Draw each vehicle's routes
        for veh, route_segments in vehicle_routes.items():
            color = get_color_from_string(veh)
            for segment in route_segments:
                if len(segment) > 1:  # Draw only if segment has at least 2 points
                    # Convert coordinates back to [lat, lon] format
                    coordinates = [[point[0], point[1]] for point in segment]
                    folium.PolyLine(coordinates, weight=2, color=color, opacity=0.8, popup=f"Vehicle: {veh}").add_to(m)
    if "points" in viz_type:
        # Check if number of points exceeds recommended limit
        if len(vehicle_data) > 20000:
            logger.warning(f"Large number of points ({len(vehicle_data)}) may impact performance")
        # Define color gradient for acceleration
        colormap = branca.colormap.LinearColormap(
            colors=["red", "yellow", "gray", "green", "blue"], vmin=-2.0, vmax=2.0
        )
        m.add_child(colormap)  # Add color scale to map

        # Add points to map
        for data in vehicle_data:
            acc = data["acc"] if data["acc"] is not None else 0
            folium.CircleMarker(
                location=[data["lat"], data["lon"]],
                radius=1.0,
                color=colormap(acc),
                fill=True,
                popup=f"Lat: {data['lat']:.6f}, Lon: {data['lon']:.6f}, Acc: {acc:.2f}, Veh: {data['veh']}",
            ).add_to(m)

    return m


def main():
    """Main function to process MQTT data and create visualization."""
    args = parse_args()

    if not args.log:
        logger.setLevel(logging.INFO)

    # Collect vehicle data
    vehicle_data = collect_vehicle_data(args.files, args.limit)

    if not vehicle_data:
        logger.error("No valid vehicle data found in input files")
        return

    # Create and save map
    m = create_map(vehicle_data, args.type)
    m.save(args.output)
    logger.info(f"Map saved to {args.output}")


if __name__ == "__main__":
    main()
