import folium
# from microbit import *
import time
import json

# 1. Initialize the map
map_center = [23.5, 121.0]  # Taiwan's approximate coordinates
map_zoom = 8
map_object = folium.Map(location=map_center, zoom_start=map_zoom)

# 2. Micro:bit Data Collection
def collect_data():
    """
    Collects random temperature and light levels for demonstration.
    """
    import random
    data = []
    for _ in range(5):  # Collect data 5 times as an example
        temp_val = 20 + random.randint(0, 10)
        light_val = random.randint(0, 100)
        data.append({"temperature": temp_val, "light": light_val})
        time.sleep(2)  # Wait 2 seconds before the next reading
    return data

# 3. Analyze data for truffle suitability
def analyze_data(data):
    """
    Matches the data with known truffle conditions and returns suitable locations.
    """
    suitable_locations = []
    for entry in data:
        if 20 <= entry["temperature"] <= 25 and entry["light"] <= 50:
            # Conditions for truffle growth based on provided documents
            suitable_locations.append(entry)
    return suitable_locations

# 4. Mark Truffle Locations on the Map
def mark_locations(locations, map_obj):
    """
    Marks suitable truffle growth locations on a map.
    """
    for i, loc in enumerate(locations):
        folium.Marker(
            location=[map_center[0] + 0.01 * i, map_center[1] + 0.01 * i],
            popup=f"Temperature: {loc['temperature']}°C, Light: {loc['light']}",
            icon=folium.Icon(color="green")
        ).add_to(map_obj)

# Main Execution
if __name__ == "__main__":
    print("Collecting data from Micro:bit...")
    collected_data = collect_data()
    print("Data collected:", collected_data)

    print("Analyzing data for truffle growth...")
    suitable_locations = analyze_data(collected_data)
    print("Suitable locations:", suitable_locations)

    print("Marking suitable locations on the map...")
    mark_locations(suitable_locations, map_object)

    # Save map to an HTML file
    map_object.save("truffle_map.html")
    print("Map saved as 'truffle_map.html'. Open it in a browser to view.")
    print("Done!")
