import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import math


class ParkingFinderService:
    """
    Core parking space finder engine
    Implements composite scoring: 0.5*availability + 0.3*distance + 0.2*cost
    """
    
    def __init__(self, ml_model=None):
        self.ml_model = ml_model
        self.parking_types = [
            'free_street', 'paid_street', 'garage_free', 'garage_paid',
            'reserved', 'handicapped', 'truck_parking', 'seasonal', 'lot_parking'
        ]
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate Haversine distance between two coordinates in kilometers
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
    
    def predict_availability(self, spot_features: pd.DataFrame) -> np.ndarray:
        """
        Predict parking spot availability using ML model
        Returns probability of spot being available (1 - occupancy_prob)
        """
        if self.ml_model is None or not hasattr(self.ml_model, 'predict_proba'):
            return np.ones(len(spot_features)) * 0.5
        
        try:
            occupancy_proba = self.ml_model.predict_proba(spot_features)[:, 1]
            availability_proba = 1 - occupancy_proba
            return availability_proba
        except Exception as e:
            print(f"Prediction error: {e}")
            return np.ones(len(spot_features)) * 0.5
    
    def calculate_composite_score(self, spots_df: pd.DataFrame, 
                                  user_lat: float, user_lon: float,
                                  max_distance_km: float = 5.0) -> pd.DataFrame:
        """
        Calculate composite score for each parking spot
        Score = 0.5*availability + 0.3*distance + 0.2*cost
        
        Higher score = better spot
        """
        spots_df = spots_df.copy()
        
        availability_scores = self.predict_availability(spots_df)
        spots_df['availability_score'] = availability_scores
        
        distances = []
        for _, row in spots_df.iterrows():
            dist = self.calculate_distance(user_lat, user_lon, row['latitude'], row['longitude'])
            distances.append(dist)
        
        spots_df['distance_km'] = distances
        spots_df['distance_score'] = 1 - (spots_df['distance_km'] / max_distance_km)
        spots_df['distance_score'] = spots_df['distance_score'].clip(0, 1)
        
        max_rate = spots_df['hourly_rate'].max()
        if max_rate > 0:
            spots_df['cost_score'] = 1 - (spots_df['hourly_rate'] / max_rate)
        else:
            spots_df['cost_score'] = 1.0
        
        spots_df['composite_score'] = (
            0.5 * spots_df['availability_score'] +
            0.3 * spots_df['distance_score'] +
            0.2 * spots_df['cost_score']
        )
        
        return spots_df
    
    def find_optimal_spots(self, spots_df: pd.DataFrame, 
                          user_lat: float, user_lon: float,
                          max_distance_km: float = 5.0,
                          top_n: int = 10,
                          spot_type_filter: Optional[List[str]] = None,
                          max_hourly_rate: Optional[float] = None) -> pd.DataFrame:
        """
        Find optimal parking spots based on composite scoring
        
        Filters:
        - Distance threshold
        - Spot type
        - Maximum hourly rate
        
        Returns top N spots sorted by composite score
        """
        filtered_df = spots_df.copy()
        
        if spot_type_filter:
            filtered_df = filtered_df[filtered_df['spot_type'].isin(spot_type_filter)]
        
        if max_hourly_rate is not None:
            filtered_df = filtered_df[filtered_df['hourly_rate'] <= max_hourly_rate]
        
        if len(filtered_df) == 0:
            return pd.DataFrame()
        
        scored_df = self.calculate_composite_score(filtered_df, user_lat, user_lon, max_distance_km)
        
        scored_df = scored_df[scored_df['distance_km'] <= max_distance_km]
        
        result_df = scored_df.nlargest(top_n, 'composite_score')
        
        result_df = result_df.sort_values('composite_score', ascending=False)
        
        return result_df
    
    def get_zone_statistics(self, spots_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Calculate parking statistics by zone
        Returns occupancy rates, average costs, and availability
        """
        zone_stats = {}
        
        for zone in spots_df['zone'].unique():
            zone_data = spots_df[spots_df['zone'] == zone]
            
            total_spots = len(zone_data)
            occupied_spots = zone_data['occupied'].sum()
            occupancy_rate = occupied_spots / total_spots if total_spots > 0 else 0
            
            avg_rate = zone_data['hourly_rate'].mean()
            available_spots = total_spots - occupied_spots
            
            zone_stats[zone] = {
                'total_spots': int(total_spots),
                'occupied_spots': int(occupied_spots),
                'available_spots': int(available_spots),
                'occupancy_rate': float(occupancy_rate),
                'average_hourly_rate': float(avg_rate),
                'spot_types': zone_data['spot_type'].value_counts().to_dict()
            }
        
        return zone_stats
    
    def get_real_time_recommendations(self, spots_df: pd.DataFrame,
                                     user_lat: float, user_lon: float,
                                     duration_hours: float = 2.0,
                                     preferences: Optional[Dict] = None) -> List[Dict]:
        """
        Get real-time parking recommendations based on user preferences
        
        Preferences:
        - max_distance: Maximum walking distance in km
        - max_cost: Maximum hourly rate
        - spot_types: Preferred spot types
        - accessibility: Require handicapped accessibility
        """
        preferences = preferences or {}
        
        max_distance = preferences.get('max_distance', 2.0)
        max_cost = preferences.get('max_cost', None)
        spot_types = preferences.get('spot_types', None)
        require_accessibility = preferences.get('accessibility', False)
        
        if require_accessibility:
            spots_df = spots_df[spots_df['spot_type'] == 'handicapped']
        
        optimal_spots = self.find_optimal_spots(
            spots_df, user_lat, user_lon,
            max_distance_km=max_distance,
            top_n=5,
            spot_type_filter=spot_types,
            max_hourly_rate=max_cost
        )
        
        recommendations = []
        
        for _, row in optimal_spots.iterrows():
            total_cost = row['hourly_rate'] * duration_hours
            
            walking_time_minutes = (row['distance_km'] * 1000) / 80
            
            recommendation = {
                'spot_id': row['spot_id'],
                'spot_type': row['spot_type'],
                'zone': row['zone'],
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'distance_km': float(row['distance_km']),
                'walking_time_minutes': int(walking_time_minutes),
                'hourly_rate': float(row['hourly_rate']),
                'estimated_cost': float(total_cost),
                'availability_probability': float(row['availability_score']),
                'composite_score': float(row['composite_score']),
                'confidence': 'high' if row['availability_score'] > 0.75 else 'medium' if row['availability_score'] > 0.5 else 'low'
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def analyze_parking_patterns(self, historical_df: pd.DataFrame) -> Dict:
        """
        Analyze historical parking patterns
        Returns peak hours, occupancy trends, and revenue insights
        """
        peak_hours = historical_df.groupby('hour')['occupancy_rate'].mean().nlargest(5)
        
        busiest_zones = historical_df.groupby('zone')['occupancy_rate'].mean().nlargest(5)
        
        weekday_pattern = historical_df[historical_df['day_of_week'] < 5].groupby('hour')['occupancy_rate'].mean()
        weekend_pattern = historical_df[historical_df['day_of_week'] >= 5].groupby('hour')['occupancy_rate'].mean()
        
        total_revenue = historical_df['revenue_per_hour'].sum()
        avg_turnover = historical_df['turnover_rate'].mean()
        
        analysis = {
            'peak_hours': peak_hours.to_dict(),
            'busiest_zones': busiest_zones.to_dict(),
            'weekday_pattern': weekday_pattern.to_dict(),
            'weekend_pattern': weekend_pattern.to_dict(),
            'total_revenue': float(total_revenue),
            'average_turnover_rate': float(avg_turnover),
            'high_demand_periods': self._identify_high_demand(historical_df)
        }
        
        return analysis
    
    def _identify_high_demand(self, df: pd.DataFrame, threshold: float = 0.8) -> List[Dict]:
        """
        Identify high-demand periods where occupancy exceeds threshold
        """
        high_demand = df[df['occupancy_rate'] > threshold]
        
        patterns = []
        for zone in high_demand['zone'].unique():
            zone_data = high_demand[high_demand['zone'] == zone]
            common_hours = zone_data['hour'].value_counts().head(3)
            
            for hour, count in common_hours.items():
                patterns.append({
                    'zone': zone,
                    'hour': int(hour),
                    'frequency': int(count),
                    'avg_occupancy': float(zone_data[zone_data['hour'] == hour]['occupancy_rate'].mean())
                })
        
        return sorted(patterns, key=lambda x: x['frequency'], reverse=True)[:10]