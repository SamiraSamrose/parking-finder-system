import math
from datetime import datetime
from typing import Tuple, Optional
import hashlib


def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate Haversine distance between two coordinates in kilometers
    Uses Earth radius of 6371 km
    """
    R = 6371.0
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    
    return distance


def format_currency(amount: float, currency: str = 'USD') -> str:
    """
    Format amount as currency string
    Returns formatted string like $12.50
    """
    if currency == 'USD':
        return f"${amount:.2f}"
    elif currency == 'EUR':
        return f"€{amount:.2f}"
    elif currency == 'GBP':
        return f"£{amount:.2f}"
    else:
        return f"{amount:.2f} {currency}"


def format_duration(minutes: int) -> str:
    """
    Format duration in minutes to human-readable string
    Returns string like 2h 30m or 45m
    """
    if minutes < 60:
        return f"{minutes}m"
    
    hours = minutes // 60
    remaining_minutes = minutes % 60
    
    if remaining_minutes == 0:
        return f"{hours}h"
    
    return f"{hours}h {remaining_minutes}m"


def generate_spot_id(latitude: float, longitude: float, spot_type: str) -> str:
    """
    Generate unique spot ID from coordinates and type
    Returns hash-based ID like SPOT_a3f9c2d1
    """
    data_string = f"{latitude:.6f}_{longitude:.6f}_{spot_type}"
    hash_digest = hashlib.md5(data_string.encode()).hexdigest()[:8]
    
    return f"SPOT_{hash_digest}"


def validate_coordinates(latitude: float, longitude: float) -> Tuple[bool, Optional[str]]:
    """
    Validate geographic coordinates
    Returns tuple of (is_valid, error_message)
    """
    if not isinstance(latitude, (int, float)):
        return False, "Latitude must be a number"
    
    if not isinstance(longitude, (int, float)):
        return False, "Longitude must be a number"
    
    if latitude < -90 or latitude > 90:
        return False, "Latitude must be between -90 and 90"
    
    if longitude < -180 or longitude > 180:
        return False, "Longitude must be between -180 and 180"
    
    return True, None


def parse_datetime(datetime_string: str, format_string: str = '%Y-%m-%d %H:%M:%S') -> Optional[datetime]:
    """
    Parse datetime string to datetime object
    Returns datetime object or None if parsing fails
    """
    try:
        return datetime.strptime(datetime_string, format_string)
    except ValueError:
        try:
            return datetime.fromisoformat(datetime_string.replace('Z', '+00:00'))
        except ValueError:
            return None


def calculate_walking_time(distance_km: float, walking_speed_kmh: float = 4.8) -> int:
    """
    Calculate walking time in minutes from distance
    Default walking speed: 4.8 km/h (80 m/min)
    """
    if distance_km <= 0:
        return 0
    
    time_hours = distance_km / walking_speed_kmh
    time_minutes = time_hours * 60
    
    return int(math.ceil(time_minutes))


def calculate_parking_cost(hourly_rate: float, duration_minutes: int) -> float:
    """
    Calculate total parking cost
    Rounds up to nearest hour for billing
    """
    if hourly_rate <= 0 or duration_minutes <= 0:
        return 0.0
    
    duration_hours = math.ceil(duration_minutes / 60.0)
    
    total_cost = hourly_rate * duration_hours
    
    return round(total_cost, 2)


def get_peak_hours() -> list:
    """
    Get typical peak parking hours
    Returns list of hours (0-23)
    """
    morning_peak = list(range(7, 10))
    evening_peak = list(range(17, 20))
    
    return morning_peak + evening_peak


def is_peak_hour(hour: int) -> bool:
    """
    Check if given hour is peak parking time
    Peak hours: 7-9 AM and 5-7 PM
    """
    return hour in get_peak_hours()


def categorize_occupancy_rate(occupancy: float) -> str:
    """
    Categorize occupancy rate into levels
    Returns: low, medium, high, critical
    """
    if occupancy < 0.3:
        return 'low'
    elif occupancy < 0.6:
        return 'medium'
    elif occupancy < 0.85:
        return 'high'
    else:
        return 'critical'