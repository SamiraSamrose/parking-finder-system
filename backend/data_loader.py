import pandas as pd
import numpy as np
import gzip
import requests
from io import BytesIO
from datetime import datetime, timedelta
from typing import Tuple, Dict
import os


class DataLoader:
    """Load and preprocess traffic and parking datasets"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.metro_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
        
        self.weather_impact_map = {
            'Clear': 0.1,
            'Clouds': 0.2,
            'Rain': 0.5,
            'Snow': 0.7,
            'Mist': 0.3,
            'Drizzle': 0.4,
            'Thunderstorm': 0.8,
            'Fog': 0.6
        }
    
    def load_metro_traffic_data(self) -> pd.DataFrame:
        """
        Load Metro Interstate Traffic Dataset from UCI ML Repository
        Features: traffic_volume, temp, rain_1h, snow_1h, clouds_all, weather_main, date_time
        """
        cache_file = os.path.join(self.data_dir, "metro_traffic.csv")
        
        if os.path.exists(cache_file):
            print(f"Loading cached traffic data from {cache_file}")
            df = pd.read_csv(cache_file)
        else:
            print(f"Downloading traffic data from {self.metro_url}")
            try:
                response = requests.get(self.metro_url, timeout=30)
                response.raise_for_status()
                
                with gzip.open(BytesIO(response.content), 'rt') as f:
                    df = pd.read_csv(f)
                
                df.to_csv(cache_file, index=False)
                print(f"Cached traffic data to {cache_file}")
            except Exception as e:
                print(f"Error downloading data: {e}")
                print("Generating synthetic traffic data")
                df = self._generate_synthetic_traffic()
        
        df = self._preprocess_traffic_data(df)
        
        return df
    
    def _generate_synthetic_traffic(self) -> pd.DataFrame:
        """Generate synthetic traffic data if download fails"""
        np.random.seed(42)
        
        date_range = pd.date_range(
            start='2023-01-01',
            end='2024-12-31',
            freq='5min'
        )
        
        n_records = len(date_range)
        
        df = pd.DataFrame({
            'date_time': date_range,
            'traffic_volume': np.random.randint(1000, 7000, n_records),
            'temp': np.random.uniform(-10, 35, n_records),
            'rain_1h': np.random.exponential(0.5, n_records),
            'snow_1h': np.random.exponential(0.1, n_records),
            'clouds_all': np.random.randint(0, 100, n_records),
            'weather_main': np.random.choice(
                list(self.weather_impact_map.keys()),
                n_records,
                p=[0.3, 0.3, 0.15, 0.05, 0.1, 0.05, 0.03, 0.02]
            )
        })
        
        return df
    
    def _preprocess_traffic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess traffic data with feature engineering
        - DateTime parsing
        - Temporal features
        - Weather impact quantification
        """
        df['date_time'] = pd.to_datetime(df['date_time'])
        
        df['hour'] = df['date_time'].dt.hour
        df['day_of_week'] = df['date_time'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['date_time'].dt.month
        df['day_of_year'] = df['date_time'].dt.dayofyear
        
        df['weather_impact'] = df['weather_main'].map(self.weather_impact_map).fillna(0.2)
        
        df['temp_normalized'] = (df['temp'] - df['temp'].mean()) / df['temp'].std()
        df['traffic_normalized'] = df['traffic_volume'] / df['traffic_volume'].max()
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def generate_parking_data(self, traffic_df: pd.DataFrame, n_spots: int = 50000) -> pd.DataFrame:
        """
        Generate parking sensor data correlated with traffic patterns
        occupancy_prob = 0.3 + (0.5 × traffic_factor × time_factor) + (0.1 × weather_factor)
        """
        np.random.seed(42)
        
        spot_types = ['free_street', 'paid_street', 'garage_free', 'garage_paid', 
                      'reserved', 'handicapped', 'truck_parking', 'seasonal', 'lot_parking']
        
        zones = ['downtown', 'midtown', 'uptown', 'waterfront', 'airport', 
                 'university', 'hospital', 'shopping', 'residential', 'industrial']
        
        records = []
        
        for i in range(n_spots):
            timestamp_idx = np.random.randint(0, len(traffic_df))
            traffic_row = traffic_df.iloc[timestamp_idx]
            
            traffic_factor = traffic_row['traffic_normalized']
            
            hour = traffic_row['hour']
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                time_factor = 1.0
            elif 10 <= hour <= 16:
                time_factor = 0.7
            else:
                time_factor = 0.3
            
            weather_factor = traffic_row['weather_impact']
            
            occupancy_prob = np.clip(
                0.3 + (0.5 * traffic_factor * time_factor) + (0.1 * weather_factor),
                0, 1
            )
            
            occupied = np.random.random() < occupancy_prob
            
            spot_type = np.random.choice(spot_types)
            zone = np.random.choice(zones)
            
            hourly_rate = 0.0 if 'free' in spot_type else np.random.uniform(2, 25)
            
            lat = 37.7749 + np.random.uniform(-0.1, 0.1)
            lon = -122.4194 + np.random.uniform(-0.1, 0.1)
            
            records.append({
                'spot_id': f"SPOT_{i:05d}",
                'timestamp': traffic_row['date_time'],
                'occupied': int(occupied),
                'spot_type': spot_type,
                'zone': zone,
                'hourly_rate': round(hourly_rate, 2),
                'latitude': round(lat, 6),
                'longitude': round(lon, 6),
                'traffic_volume': traffic_row['traffic_volume'],
                'temp': traffic_row['temp'],
                'weather_main': traffic_row['weather_main'],
                'weather_impact': traffic_row['weather_impact'],
                'hour': hour,
                'day_of_week': traffic_row['day_of_week'],
                'is_weekend': traffic_row['is_weekend']
            })
        
        parking_df = pd.DataFrame(records)
        
        return parking_df
    
    def generate_curb_regulations(self, n_segments: int = 2000) -> pd.DataFrame:
        """Generate curb regulation data"""
        np.random.seed(42)
        
        regulation_types = ['metered', 'time_limited', 'permit_only', 'no_parking', 
                           'loading_zone', 'residential', 'commercial', 'handicapped']
        
        vehicle_types = ['passenger', 'commercial', 'handicapped', 'electric', 'motorcycle']
        
        records = []
        
        for i in range(n_segments):
            reg_type = np.random.choice(regulation_types)
            
            if reg_type == 'metered':
                time_restrictions = "Mon-Fri 8AM-6PM"
                max_stay = np.random.choice([60, 120, 240])
            elif reg_type == 'time_limited':
                time_restrictions = "Mon-Sun 7AM-10PM"
                max_stay = np.random.choice([15, 30, 60, 120])
            elif reg_type == 'no_parking':
                time_restrictions = "Always"
                max_stay = 0
            else:
                time_restrictions = "Mon-Sun 24/7"
                max_stay = np.random.choice([60, 120, 240, 480])
            
            allowed_vehicles = np.random.choice(vehicle_types, size=np.random.randint(1, 4), replace=False).tolist()
            
            lat = 37.7749 + np.random.uniform(-0.1, 0.1)
            lon = -122.4194 + np.random.uniform(-0.1, 0.1)
            
            records.append({
                'segment_id': f"SEG_{i:04d}",
                'regulation_type': reg_type,
                'time_restrictions': time_restrictions,
                'max_stay_minutes': max_stay,
                'vehicle_types': ','.join(allowed_vehicles),
                'latitude': round(lat, 6),
                'longitude': round(lon, 6)
            })
        
        return pd.DataFrame(records)
    
    def generate_historical_patterns(self) -> pd.DataFrame:
        """
        Generate historical parking patterns aggregated by zone and hour
        17,520 records (24 hours × 730 days)
        """
        np.random.seed(42)
        
        zones = ['downtown', 'midtown', 'uptown', 'waterfront', 'airport', 
                 'university', 'hospital', 'shopping', 'residential', 'industrial']
        
        date_range = pd.date_range(start='2023-01-01', end='2024-12-31', freq='H')
        
        records = []
        
        for timestamp in date_range:
            for zone in zones:
                hour = timestamp.hour
                day_of_week = timestamp.dayofweek
                
                if zone == 'downtown':
                    if 7 <= hour <= 19:
                        base_occupancy = 0.75
                    else:
                        base_occupancy = 0.3
                elif zone == 'residential':
                    if 18 <= hour <= 8:
                        base_occupancy = 0.85
                    else:
                        base_occupancy = 0.4
                elif zone == 'university':
                    if day_of_week < 5 and 8 <= hour <= 17:
                        base_occupancy = 0.9
                    else:
                        base_occupancy = 0.2
                else:
                    base_occupancy = 0.5
                
                occupancy_rate = np.clip(base_occupancy + np.random.uniform(-0.15, 0.15), 0, 1)
                turnover_rate = np.random.uniform(0.1, 0.5)
                revenue_per_hour = occupancy_rate * np.random.uniform(10, 50)
                
                records.append({
                    'timestamp': timestamp,
                    'zone': zone,
                    'hour': hour,
                    'day_of_week': day_of_week,
                    'occupancy_rate': round(occupancy_rate, 3),
                    'turnover_rate': round(turnover_rate, 3),
                    'revenue_per_hour': round(revenue_per_hour, 2)
                })
        
        return pd.DataFrame(records)
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load and return all datasets"""
        print("Loading Metro Interstate Traffic Dataset...")
        traffic_df = self.load_metro_traffic_data()
        print(f"Loaded {len(traffic_df)} traffic records")
        
        print("Generating parking sensor data...")
        parking_df = self.generate_parking_data(traffic_df)
        print(f"Generated {len(parking_df)} parking records")
        
        print("Generating curb regulations...")
        curb_df = self.generate_curb_regulations()
        print(f"Generated {len(curb_df)} curb segments")
        
        print("Generating historical patterns...")
        historical_df = self.generate_historical_patterns()
        print(f"Generated {len(historical_df)} historical records")
        
        datasets = {
            'traffic': traffic_df,
            'parking': parking_df,
            'curb': curb_df,
            'historical': historical_df
        }
        
        for name, df in datasets.items():
            cache_file = os.path.join(self.data_dir, f"{name}_data.csv")
            df.to_csv(cache_file, index=False)
            print(f"Cached {name} data to {cache_file}")
        
        return datasets