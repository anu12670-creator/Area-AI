
import math

# 1. API Headers
HEADERS = {
    "User-Agent": "AreaAI/1.0 (Neighborhood Analysis Project)"
}


# 2. Calculate distance between two geographical locations
def calculate_distance(lat1, lon1, lat2, lon2):

    earth_radius = 6371  # Earth radius in kilometers

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius * c


# 3. Get facility name
def get_facility_name(tags):

    if not isinstance(tags, dict):
        return "Unknown Facility"

    return tags.get("name", "Unnamed Facility")


# 4. Classify facility

def classify_facility(tags):

    if not isinstance(tags, dict):
        return "Other"

    amenity = tags.get("amenity", "")
    shop = tags.get("shop", "")
    healthcare = tags.get("healthcare", "")
    leisure = tags.get("leisure", "")
    tourism = tags.get("tourism", "")
    highway = tags.get("highway", "")
    railway = tags.get("railway", "")

    # Healthcare
    if amenity in ["hospital", "clinic", "doctors", "pharmacy"] or healthcare:
        return "Healthcare"

    # Education
    elif amenity in ["school", "college", "university", "kindergarten"]:
        return "Education"

    # Shopping
    elif shop:
        return "Shopping"

    # Food
    elif amenity in ["restaurant", "cafe", "fast_food", "food_court"]:
        return "Food"

    # Finance
    elif amenity in ["bank", "atm"]:
        return "Finance"

    # Transport
    elif amenity in ["bus_station", "bus_stop", "taxi"] or railway in ["station", "halt"]:
        return "Transport"

    # Religious places
    elif amenity in ["place_of_worship"]:
        return "Religious Places"

    # Recreation
    elif leisure in ["park", "garden", "playground", "sports_centre"]:
        return "Recreation"

    # Tourism
    elif tourism:
        return "Tourism"

    # Roads
    elif highway in ["primary", "secondary", "tertiary", "trunk"]:
        return "Roads"

    else:
        return "Other"


   