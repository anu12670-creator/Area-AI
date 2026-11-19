import re
import time
from urllib.parse import urlparse, parse_qs, unquote

import requests
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation

from function_definition import (
    HEADERS,
    calculate_distance,
    get_facility_name,
    classify_facility,
)
from ai_report import generate_neighbourhood_report

st.set_page_config(page_title="Area AI", page_icon="📍", layout="wide")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
VALHALLA_MATRIX_URL = "https://valhalla1.openstreetmap.de/sources_to_targets"
VALHALLA_ROUTE_URL = "https://valhalla1.openstreetmap.de/route"
OVERPASS_URLS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
GOOGLE_HOSTS = {
    "google.com", "www.google.com", "maps.google.com",
    "google.co.in", "www.google.co.in", "maps.app.goo.gl",
    "goo.gl",
}


def is_google_host(host):
    if not host:
        return False
    host = host.lower().split(":")[0]
    return host in GOOGLE_HOSTS or host.endswith(".google.com") or host.endswith(".google.co.in")


def coordinates_from_text(text):
    """Find a latitude/longitude pair in text, including common Google Maps URL formats."""
    patterns = [
        r"!3d(-?\d+(?:\.\d+)?)!4d(-?\d+(?:\.\d+)?)",
        r"@(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)",
        r"(?:[?&](?:q|query|ll|center)=)(-?\d+(?:\.\d+)?)[,%2C ]+(-?\d+(?:\.\d+)?)",
        r"(-?\d{1,2}\.\d{3,})\s*,\s*(-?\d{1,3}\.\d{3,})",
    ]
    decoded = unquote(text)
    for pattern in patterns:
        match = re.search(pattern, decoded, flags=re.IGNORECASE)
        if match:
            lat, lon = float(match.group(1)), float(match.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
    return None


def resolve_google_short_link(url):
    """Follow a small number of redirects, allowing only Google-owned hosts."""
    current = url
    for _ in range(5):
        parsed = urlparse(current)
        if parsed.scheme not in ("http", "https") or not is_google_host(parsed.hostname):
            raise ValueError("Please use a Google Maps link.")
        response = requests.get(
            current,
            headers=HEADERS,
            allow_redirects=False,
            timeout=12,
        )
        if response.status_code in (301, 302, 303, 307, 308):
            next_url = response.headers.get("Location")
            if not next_url:
                break
            if next_url.startswith("/"):
                next_url = f"{parsed.scheme}://{parsed.netloc}{next_url}"
            current = next_url
            continue
        break
    return current


@st.cache_data(ttl=3600, show_spinner=False)
def geocode_address(address):
    response = requests.get(
        NOMINATIM_URL,
        params={"q": address, "format": "json", "limit": 1},
        headers=HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        return None
    return float(results[0]["lat"]), float(results[0]["lon"]), results[0].get("display_name", address)


def resolve_location(value):
    value = value.strip()
    if not value:
        raise ValueError("Enter an address, Google Maps link, or latitude, longitude.")

    coords = coordinates_from_text(value)
    if coords:
        return coords[0], coords[1], "Coordinates"

    if value.lower().startswith(("http://", "https://")):
        parsed = urlparse(value)
        if not is_google_host(parsed.hostname):
            raise ValueError("Only Google Maps links are supported. You can also paste an address or coordinates.")
        expanded_url = resolve_google_short_link(value)
        coords = coordinates_from_text(expanded_url)
        if coords:
            return coords[0], coords[1], "Google Maps location"
        # Some Google Maps links contain a place name but no coordinates.
        parsed_expanded = urlparse(expanded_url)
        query = parse_qs(parsed_expanded.query).get("q", [""])[0]
        if not query:
            query = unquote(parsed_expanded.path.split("/place/")[-1].split("/")[0]).replace("+", " ")
        if query:
            result = geocode_address(query)
            if result:
                return result
        raise ValueError("I couldn't read a location from this Google Maps link. Try copying the full Maps link or paste the address.")

    result = geocode_address(value)
    if not result:
        raise ValueError("Location not found. Try a more complete address or paste its Google Maps link.")
    return result


def decode_valhalla_shape(encoded, precision=6):
    """Decode Valhalla's encoded polyline6 into (latitude, longitude) pairs."""
    coordinates = []
    index = 0
    lat = 0
    lon = 0
    factor = 10 ** precision

    while index < len(encoded):
        values = []
        for _ in range(2):
            result = 0
            shift = 0
            while True:
                byte = ord(encoded[index]) - 63
                index += 1
                result |= (byte & 0x1F) << shift
                shift += 5
                if byte < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else (result >> 1)
            values.append(delta)

        lat += values[0]
        lon += values[1]
        coordinates.append((lat / factor, lon / factor))

    return coordinates


@st.cache_data(ttl=900, show_spinner=False)
def get_route_shape(start_lat, start_lon, end_lat, end_lon, travel_mode):
    """Request the actual route geometry from Valhalla for drawing on the map."""
    costing = "pedestrian" if travel_mode == "Walking" else "auto"
    payload = {
        "locations": [
            {"lat": float(start_lat), "lon": float(start_lon)},
            {"lat": float(end_lat), "lon": float(end_lon)},
        ],
        "costing": costing,
        "units": "kilometers",
        "shape_format": "polyline6",
    }
    response = requests.post(
        VALHALLA_ROUTE_URL,
        json=payload,
        headers={**HEADERS, "Content-Type": "application/json"},
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    legs = data.get("trip", {}).get("legs", [])
    if not legs or not legs[0].get("shape"):
        raise RuntimeError("The routing service did not return a route line.")
    return decode_valhalla_shape(legs[0]["shape"])


@st.cache_data(ttl=900, show_spinner=False)
def get_road_distances(source_lat, source_lon, target_points, travel_mode):
    """Get road/path distances from one source to many targets using Valhalla."""
    if not target_points:
        return []

    costing = "pedestrian" if travel_mode == "Walking" else "auto"
    results = []

    # Send manageable batches to the routing matrix service.
    for start in range(0, len(target_points), 40):
        batch = target_points[start:start + 40]
        payload = {
            "sources": [{"lat": float(source_lat), "lon": float(source_lon)}],
            "targets": [
                {"lat": float(point[0]), "lon": float(point[1])}
                for point in batch
            ],
            "costing": costing,
            "units": "kilometers",
            "verbose": True,
        }
        response = requests.post(
            VALHALLA_MATRIX_URL,
            json=payload,
            headers={**HEADERS, "Content-Type": "application/json", "X-Client-Id": "area-ai-rental-decision-app"},
            timeout=45,
        )
        response.raise_for_status()
        matrix = response.json().get("sources_to_targets", [])
        row = matrix[0] if matrix else []
        for index in range(len(batch)):
            distance = row[index].get("distance") if index < len(row) else None
            results.append(float(distance) if distance is not None else None)
    return results



@st.cache_data(ttl=900, show_spinner=False)
def get_nearby_facilities(latitude, longitude, radius, travel_mode):
    """
    Fetch nearby facilities using multiple Overpass servers.
    Keep facilities even when Valhalla routing is unavailable.
    """

    radius_m = max(250, min(int(radius), 1000))

    query = f"""
    [out:json][timeout:40];
    (
      node(around:{radius_m},{latitude},{longitude})[amenity~"^(hospital|clinic|doctors|pharmacy|school|college|university|bus_station|restaurant|cafe|place_of_worship|bank|atm|library)$"];
      way(around:{radius_m},{latitude},{longitude})[amenity~"^(hospital|clinic|doctors|pharmacy|school|college|university|bus_station|restaurant|cafe|place_of_worship|bank|atm|library)$"];

      node(around:{radius_m},{latitude},{longitude})[shop];
      way(around:{radius_m},{latitude},{longitude})[shop];

      node(around:{radius_m},{latitude},{longitude})[leisure~"^(park|playground|sports_centre|fitness_centre)$"];
      way(around:{radius_m},{latitude},{longitude})[leisure~"^(park|playground|sports_centre|fitness_centre)$"];

      node(around:{radius_m},{latitude},{longitude})[public_transport~"^(platform|stop_position)$"];
      node(around:{radius_m},{latitude},{longitude})[railway~"^(station|halt|subway_entrance)$"];
    );
    out center tags;
    """

    elements = None
    last_error = None

    # Try multiple Overpass servers
    for endpoint in OVERPASS_URLS:
        try:
            response = requests.post(
                endpoint,
                data={"data": query},
                headers=HEADERS,
                timeout=(10, 60)
            )

            response.raise_for_status()

            data = response.json()

            if "elements" not in data:
                raise ValueError("Invalid Overpass response")

            elements = data["elements"]
            break

        except Exception as error:
            last_error = error
            continue

    if elements is None:
        raise RuntimeError(
            "All OpenStreetMap servers failed. "
            f"Last error: {last_error}"
        )

    # Convert results into rows
    rows = []
    seen = set()

    for item in elements:

        osm_id = f'{item.get("type")}/{item.get("id")}'

        if osm_id in seen:
            continue

        seen.add(osm_id)

        tags = item.get("tags", {})
        center = item.get("center", {})

        lat = item.get("lat", center.get("lat"))
        lon = item.get("lon", center.get("lon"))

        if lat is None or lon is None:
            continue

        rows.append({
            "Name": get_facility_name(tags),
            "Category": classify_facility(tags),
            "Latitude": float(lat),
            "Longitude": float(lon),
            "OSM ID": osm_id,
        })

    if not rows:
        return pd.DataFrame()

    # Try actual road/path distances
    road_distances = None

    try:
        road_distances = get_road_distances(
            latitude,
            longitude,
            [
                (row["Latitude"], row["Longitude"])
                for row in rows
            ],
            travel_mode,
        )

    except Exception:
        road_distances = None

    # Fallback to straight-line distances
    for index, row in enumerate(rows):

        distance_km = None

        if road_distances is not None and index < len(road_distances):
            distance_km = road_distances[index]

        if distance_km is not None:

            row["Distance (km)"] = round(distance_km, 3)

            if distance_km < 1:
                row["Distance"] = f"{distance_km * 1000:.0f} m"
            else:
                row["Distance"] = f"{distance_km:.2f} km"

        else:

            straight_distance = calculate_distance(
                latitude,
                longitude,
                row["Latitude"],
                row["Longitude"],
            )

            row["Distance (km)"] = round(straight_distance, 3)

            if straight_distance < 1:
                distance_text = f"{straight_distance * 1000:.0f} m"
            else:
                distance_text = f"{straight_distance:.2f} km"

            row["Distance"] = (
                f"Approx. {distance_text} (straight-line)"
            )

    return pd.DataFrame(rows).sort_values(
        "Distance (km)",
        na_position="last",
    )



def make_map(house_lat, house_lon, start=None, selected_facility=None, facilities=None, route_points=None):
    map_obj = folium.Map(location=[house_lat, house_lon], zoom_start=15, control_scale=True)
    folium.Marker(
        [house_lat, house_lon],
        tooltip="Rental house",
        popup="Selected rental house",
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
    ).add_to(map_obj)

    if start:
        folium.Marker(
            [start[0], start[1]],
            tooltip="Starting point",
            popup="Your starting point",
            icon=folium.Icon(color="blue", icon="play", prefix="fa"),
        ).add_to(map_obj)

    if route_points:
        folium.PolyLine(
            route_points,
            color="blue",
            weight=6,
            opacity=0.85,
            tooltip="Selected walking/driving route",
        ).add_to(map_obj)
        map_obj.fit_bounds(route_points, padding=(25, 25))

    if selected_facility is not None:
        folium.Marker(
            [selected_facility["Latitude"], selected_facility["Longitude"]],
            tooltip=selected_facility["Name"],
            popup=f'{selected_facility["Name"]}: {selected_facility["Distance"]} from rental house',
            icon=folium.Icon(color="green", icon="info-sign"),
        ).add_to(map_obj)

    elif facilities is not None and not facilities.empty:
        for _, row in facilities.head(100).iterrows():
            folium.CircleMarker(
                [row["Latitude"], row["Longitude"]],
                radius=5,
                color="green",
                fill=True,
                fill_opacity=0.75,
                tooltip=f'{row["Name"]} — {row["Distance"]}',
            ).add_to(map_obj)

    return map_obj


st.title("📍 Area AI")
st.subheader("AI-Powered Neighbourhood & Rental Decision System")
st.write("Explore the facilities around a rental house and check distances from your starting point.")

st.divider()
st.header("1. Select the rental house location")
house_input = st.text_input(
    "Paste an address, Google Maps link, or coordinates",
    placeholder="Example: 17.4930841, 78.4054408",
    key="house_input",
)
if st.button("Find rental house location", type="primary"):
    try:
        with st.spinner("Finding rental house location..."):
            lat, lon, label = resolve_location(house_input)
        st.session_state["house_location"] = (lat, lon, label)
        st.success(f"Rental house location found: {label}")
    except Exception as error:
        st.error(str(error))

if "house_location" not in st.session_state:
    st.info("Enter the rental house location above to continue.")
    st.stop()

house_lat, house_lon, house_label = st.session_state["house_location"]
st.caption(f"Rental house coordinates: {house_lat:.6f}, {house_lon:.6f}")
travel_mode = st.radio("Distance mode", ["Walking", "Driving"], horizontal=True, key="travel_mode")

st.write("You can click a different point on the map to move the rental-house pin.")
house_map = make_map(house_lat, house_lon)
map_result = st_folium(house_map, height=420, width=None, key="house_map")
if map_result and map_result.get("last_clicked"):
    clicked = map_result["last_clicked"]
    new_lat, new_lon = float(clicked["lat"]), float(clicked["lng"])
    if abs(new_lat - house_lat) > 0.000001 or abs(new_lon - house_lon) > 0.000001:
        st.session_state["house_location"] = (new_lat, new_lon, "Selected on map")
        st.rerun()

st.divider()
st.header("2. Choose your starting point")
start_choice = st.radio(
    "Where are you starting from?",
    ["Use my current location", "Enter another location"],
    horizontal=True,
)

start_location = None
if start_choice == "Use my current location":
    st.caption("Your browser will ask permission to share your location.")
    current = streamlit_geolocation()
    if current and current.get("latitude") is not None and current.get("longitude") is not None:
        start_location = (float(current["latitude"]), float(current["longitude"]))
        st.success(f"Current location received: {start_location[0]:.6f}, {start_location[1]:.6f}")
    else:
        st.info("Allow location access in your browser. Or choose 'Enter another location'.")
else:
    start_input = st.text_input(
        "Enter your starting address, Google Maps link, or coordinates",
        placeholder="Enter your current area or paste a Maps link",
        key="start_input",
    )
    if st.button("Find starting point"):
        try:
            with st.spinner("Finding starting point..."):
                start_lat, start_lon, start_label = resolve_location(start_input)
            st.session_state["start_location"] = (start_lat, start_lon, start_label)
            st.success(f"Starting point found: {start_label}")
        except Exception as error:
            st.error(str(error))
    if "start_location" in st.session_state:
        saved_start = st.session_state["start_location"]
        start_location = (saved_start[0], saved_start[1])

if start_location:
    try:
        with st.spinner(f"Calculating {travel_mode.lower()} route..."):
            route_km = get_road_distances(
                start_location[0], start_location[1], [(house_lat, house_lon)], travel_mode
            )[0]
        if route_km is None:
            st.warning("A route could not be found between these locations.")
        else:
            st.metric(f"{travel_mode} distance to rental house", f"{route_km:.2f} km")
    except Exception as error:
        st.error(f"Could not calculate the route right now: {error}")
    try:
        with st.spinner("Loading route line on the map..."):
            start_route_points = get_route_shape(
                start_location[0], start_location[1],
                house_lat, house_lon, travel_mode
            )
        st_folium(
            make_map(
                house_lat, house_lon,
                start=start_location,
                route_points=start_route_points,
            ),
            height=420,
            width=None,
            key="start_map",
        )
    except Exception as error:
        st.warning(f"Distance was calculated, but the route line could not be loaded: {error}")
        st_folium(
            make_map(house_lat, house_lon, start=start_location),
            height=350,
            width=None,
            key="start_map_without_route",
        )

st.divider()
st.header("3. Find nearby facilities")
radius = st.slider("Search radius around the rental house", min_value=250, max_value=3000, value=1500, step=250)
if st.button("Find nearby facilities", type="primary"):
    try:
        with st.spinner("Searching for nearby facilities..."):
            df = get_nearby_facilities(house_lat, house_lon, radius, travel_mode)
        st.session_state["facilities"] = df
        st.session_state["facilities_house"] = (house_lat, house_lon, travel_mode)
    except Exception as error:
        st.error(str(error))

facilities = st.session_state.get("facilities")
facilities_house = st.session_state.get("facilities_house")
if facilities is not None:
    if facilities_house != (house_lat, house_lon, travel_mode):
        st.info("The rental-house location or distance mode has changed. Search again to refresh nearby facilities.")
    elif facilities.empty:
        st.warning("No facilities were found in this radius. Try increasing the search radius.")
    else:
        if any("straight-line" in str(value) for value in facilities["Distance"]):
            st.warning("Some distances are approximate straight-line distances, not walking or driving route distances.")
        st.success(f"Found {len(facilities)} nearby places.")
        st.dataframe(
            facilities[["Name", "Category", "Distance"]],
                width="stretch",
                hide_index=True,

        )

        # Generate an AI report using only the facilities collected by Area AI.
        st.divider()
        st.header("4. AI Neighbourhood Report")
        st.write("Generate a report from the nearby facilities and route distances found above.")

        report_context = (house_lat, house_lon, travel_mode, radius)

        if st.button("Generate AI Neighbourhood Report", type="primary", key="generate_ai_report"):
            try:
                report_data = facilities[["Name", "Category", "Distance"]].to_string(index=False)

                with st.spinner("Gemini is analysing the neighbourhood..."):
                    generated_report = generate_neighbourhood_report(
                        location=house_label,
                        travel_mode=travel_mode,
                        facility_data=report_data,
                    )

                st.session_state["ai_report"] = generated_report
                st.session_state["ai_report_context"] = report_context
                st.success("Your neighbourhood report is ready.")
            except Exception as error:
                st.error(f"Could not generate the AI report: {error}")

        saved_report = st.session_state.get("ai_report")
        saved_report_context = st.session_state.get("ai_report_context")
        if saved_report and saved_report_context == report_context:
            st.subheader("Your Neighbourhood Analysis")
            st.markdown(saved_report)

        st.divider()
        st.header("5. Check the distance to an individual facility")
        facility_options = facilities.apply(
            lambda row: f'{row["Name"]} — {row["Category"]} — {row["Distance"]}',
            axis=1,
        ).tolist()
        chosen_label = st.selectbox("Select a nearby facility", facility_options)
        chosen_index = facility_options.index(chosen_label)
        selected = facilities.iloc[chosen_index].to_dict()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Selected facility", selected["Name"])
            st.write(f'**Category:** {selected["Category"]}')
        with col2:
            st.metric("Distance from rental house", selected["Distance"])
            st.caption(f"{travel_mode} route distance, or approximate straight-line distance if routing is unavailable")

        try:
            with st.spinner("Loading route line to the selected facility..."):
                facility_route_points = get_route_shape(
                    house_lat, house_lon,
                    selected["Latitude"], selected["Longitude"],
                    travel_mode,
                )
            st_folium(
                make_map(
                    house_lat, house_lon,
                    selected_facility=selected,
                    route_points=facility_route_points,
                ),
                height=420,
                width=None,
                key="facility_map",
            )
        except Exception as error:
            st.warning(f"Distance is available, but the route line could not be loaded: {error}")
            st_folium(
                make_map(house_lat, house_lon, selected_facility=selected),
                height=380,
                width=None,
                key="facility_map_without_route",
            )

        csv_data = facilities.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download nearby facilities (CSV)",
            data=csv_data,
            file_name="area_ai_nearby_facilities.csv",
            mime="text/csv",
        )

st.divider()
st.caption("Area AI uses OpenStreetMap data and Valhalla routing. Results depend on mapped roads, paths, and facility information.")
