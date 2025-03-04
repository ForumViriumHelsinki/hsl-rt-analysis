import folium
import pandas as pd
from branca.colormap import LinearColormap


def load_and_prepare_data(file_path):
    """Load and prepare HSL data for visualization."""
    df = pd.read_csv(file_path)
    return df


def get_color_for_acceleration(acc):
    """
    Generate color based on acceleration value.
    Negative acceleration (deceleration): yellow -> red
    Positive acceleration: green -> blue
    """
    if acc < 0:
        # For deceleration: interpolate between yellow and red
        intensity = min(-acc / 2.0, 1.0)  # Normalize to [0,1]
        return f"#{int(255):02x}{int(255 * (1 - intensity)):02x}{0:02x}"  # Yellow to Red
    else:
        # For acceleration: interpolate between green and blue
        intensity = min(acc / 2.0, 1.0)  # Normalize to [0,1]
        return f"#{0:02x}{int(255 * (1 - intensity)):02x}{int(255 * intensity):02x}"  # Green to Blue


def create_route_map(df):
    """Create a map visualization of the route with acceleration-based coloring."""
    # Initialize map centered on the first coordinate
    m = folium.Map(location=[df.iloc[0]["lat"], df.iloc[0]["long"]], zoom_start=15, tiles="cartodbpositron")

    # Add route segments with colors based on acceleration and popups
    for i in range(len(df) - 1):
        points = [[df.iloc[i]["lat"], df.iloc[i]["long"]], [df.iloc[i + 1]["lat"], df.iloc[i + 1]["long"]]]

        color = get_color_for_acceleration(df.iloc[i]["acc"])

        # Create popup content with time, speed and acceleration
        popup_content = f"""
            <div style='font-family: Arial; font-size: 12px;'>
                <b>Time:</b> {df.iloc[i]["tst"]}<br>
                <b>Speed:</b> {df.iloc[i]["spd"]:.1f} km/h<br>
                <b>Acceleration:</b> {df.iloc[i]["acc"]:.2f} m/s²
            </div>
        """

        # Create popup object
        popup = folium.Popup(popup_content, max_width=300)

        # Add PolyLine with popup
        folium.PolyLine(points, weight=4, color=color, opacity=0.8, popup=popup).add_to(m)

    # Add start and end markers
    folium.Marker(
        [df.iloc[0]["lat"], df.iloc[0]["long"]], popup="Lähtöpiste", icon=folium.Icon(color="green", icon="info-sign")
    ).add_to(m)

    folium.Marker(
        [df.iloc[-1]["lat"], df.iloc[-1]["long"]], popup="Päätepiste", icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    # Add two separate color maps for acceleration and deceleration
    decel_colormap = LinearColormap(
        colors=["#FF0000", "#FFFF00"],  # keltaisesta punaiseen
        vmin=-2,
        vmax=0,
        caption="Hidastuvuus (m/s²)",
    )
    accel_colormap = LinearColormap(
        colors=["#00FF00", "#0000FF"],  # vihreästä siniseen
        vmin=0,
        vmax=2,
        caption="Kiihtyvyys (m/s²)",
    )

    decel_colormap.add_to(m)
    accel_colormap.add_to(m)

    return m


def main():
    # File path
    file_path = "data/samples/hsl2015-2025-03-01-one-stop-viikki.csv"

    # Load and prepare data
    df = load_and_prepare_data(file_path)

    # Create map
    m = create_route_map(df)

    # Save map
    output_path = "reports/route_visualization.html"
    m.save(output_path)
    print(f"Kartta tallennettu: {output_path}")


if __name__ == "__main__":
    main()
