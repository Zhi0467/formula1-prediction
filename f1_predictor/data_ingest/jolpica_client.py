"""
Client for interacting with the Jolpica F1 API.
This client provides access to Formula 1 data through the Jolpica API,
which is a backwards-compatible replacement for the Ergast API.

The API provides access to:
- Race results
- Qualifying results
- Driver standings
- Constructor standings
- Lap times
- Circuit information

For more information, visit: http://api.jolpi.ca/ergast/f1/
"""
import requests
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from time import sleep

class JolpicaClient:
    """
    Client for interacting with the Jolpica/Ergast F1 API.
    Fetches race results, qualifying results, and driver/team standings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Jolpica/Ergast API client.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.base_url = "http://api.jolpi.ca/ergast/f1"
        self.api_key = config.get('api_key', None)
        self.headers = {
            'User-Agent': 'F1-Predictor/1.0',
            'Accept': 'application/json'
        }
        if self.api_key:
            self.headers['Authorization'] = f'Bearer {self.api_key}'
    
    def _make_request(self, url: str) -> Dict[str, Any]:
        """
        Make a request to the Jolpica API with rate limiting.
        """
        # Add delay between requests to respect rate limits
        sleep(0.5)  # 500ms delay between requests
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error making request to Jolpica API: {str(e)}")
    
    def get_race_results(self, season: Union[int, str], race: Optional[Union[int, str]] = None) -> pd.DataFrame:
        """
        Get race results for a specific season and optionally a specific race.
        
        Args:
            season: F1 season (year).
            race: Race number or 'last' for the most recent race.
            
        Returns:
            DataFrame with race results.
        """
        race_str = f"/{race}" if race is not None else ""
        url = f"{self.base_url}/{season}{race_str}/results.json"
        
        data = self._make_request(url)
        races_data = data.get('MRData', {}).get('RaceTable', {}).get('Races', [])
        
        if not races_data:
            return pd.DataFrame()
        
        results = []
        for race_entry in races_data: # Renamed race_data to race_entry to avoid conflict
            race_name = race_entry.get('raceName')
            circuit_info = race_entry.get('Circuit', {})
            circuit_id = circuit_info.get('circuitId') # Get circuitId
            circuit_name = circuit_info.get('circuitName')
            race_date = race_entry.get('date')
            
            for result in race_entry.get('Results', []):
                driver_data = result.get('Driver', {})
                constructor_data = result.get('Constructor', {})
                
                grid_pos = int(result.get('grid', 0))
                
                finish_time_info = result.get('Time', {})
                if finish_time_info and 'time' in finish_time_info:
                    finish_time = finish_time_info['time']
                    status = "Finished"
                else:
                    finish_time = None
                    status = result.get('status', "Unknown")
                
                results.append({
                    'season': season,
                    'race_name': race_name,
                    'circuit_id': circuit_id, # Added circuit_id
                    'circuit_name': circuit_name,
                    'race_date': race_date,
                    'driver_id': driver_data.get('driverId'),
                    'driver_code': driver_data.get('code', driver_data.get('driverId', '')[:3].upper()),
                    'driver_number': driver_data.get('permanentNumber', ''),
                    'driver_name': f"{driver_data.get('givenName', '')} {driver_data.get('familyName', '')}".strip(),
                    'team_id': constructor_data.get('constructorId'),
                    'team_name': constructor_data.get('name'),
                    'position': int(result.get('position', 0)),
                    'grid_position': grid_pos,
                    'points': float(result.get('points', 0.0)),
                    'laps': int(result.get('laps', 0)),
                    'finish_time': finish_time,
                    'status': status
                })
        
        return pd.DataFrame(results)
    
    def get_qualifying_results(self, season: Union[int, str], race: Optional[Union[int, str]] = None) -> pd.DataFrame:
        """
        Get qualifying results for a specific season and optionally a specific race.
        
        Args:
            season: F1 season (year).
            race: Race number or 'last' for the most recent race.
            
        Returns:
            DataFrame with qualifying results.
        """
        race_str = f"/{race}" if race is not None else ""
        url = f"{self.base_url}/{season}{race_str}/qualifying.json"
        
        data = self._make_request(url)
        races_data = data['MRData']['RaceTable']['Races']
        
        if not races_data:
            return pd.DataFrame()
        
        qualifying_results = []
        for race_data in races_data:
            race_name = race_data['raceName']
            circuit_name = race_data['Circuit']['circuitName']
            race_date = race_data['date']
            
            for qualifying in race_data['QualifyingResults']:
                driver_data = qualifying['Driver']
                constructor_data = qualifying['Constructor']
                
                quali_result = {
                    'season': season,
                    'race_name': race_name,
                    'circuit_name': circuit_name,
                    'race_date': race_date,
                    'driver_id': driver_data['driverId'],
                    'driver_code': driver_data.get('code', driver_data['driverId'][:3].upper()),
                    'driver_number': driver_data.get('permanentNumber', ''),
                    'driver_name': f"{driver_data['givenName']} {driver_data['familyName']}",
                    'team_id': constructor_data['constructorId'],
                    'team_name': constructor_data['name'],
                    'position': int(qualifying['position']),
                }
                
                # Add Q1, Q2, Q3 times if available
                for q in ['Q1', 'Q2', 'Q3']:
                    quali_result[q] = qualifying.get(q, None)
                
                qualifying_results.append(quali_result)
        
        return pd.DataFrame(qualifying_results)
    
    def get_driver_standings(self, season: Union[int, str], round_num: Optional[Union[int, str]] = None) -> pd.DataFrame:
        """
        Get driver standings for a specific season and optionally after a specific round.
        
        Args:
            season: F1 season (year).
            round_num: Round number or 'current' for the latest round.
            
        Returns:
            DataFrame with driver standings.
        """
        round_str = f"/{round_num}" if round_num is not None else ""
        url = f"{self.base_url}/{season}{round_str}/driverStandings.json"
        
        data = self._make_request(url)
        
        # It's good practice to check the actual structure of 'data' if errors persist
        # print(f"Debug: API response for driver standings ({season}, {round_num}): {data}") 

        standings_data = data.get('MRData', {}).get('StandingsTable', {}).get('StandingsLists', [])
        
        if not standings_data:
            return pd.DataFrame()
        
        drivers_standings = []
        for standing_list in standings_data:
            # Use .get() for season and round from standing_list as well for robustness
            current_season = standing_list.get('season', season) # Fallback to requested season
            current_round = standing_list.get('round', round_num if isinstance(round_num, (int, str)) else 0) # Fallback
            
            for standing in standing_list.get('DriverStandings', []):
                driver_data = standing.get('Driver', {})
                
                # Get constructor info
                constructors = []
                constructor_names = []
                for constructor in standing.get('Constructors', []):
                    constructors.append(constructor.get('constructorId'))
                    constructor_names.append(constructor.get('name'))
                
                drivers_standings.append({
                    'season': current_season,
                    'round': current_round,
                    # Use .get() for position, points, and wins with default 0
                    'position': int(standing.get('position', 20)), 
                    'driver_id': driver_data.get('driverId'),
                    'driver_code': driver_data.get('code', driver_data.get('driverId', '')[:3].upper()),
                    'driver_number': driver_data.get('permanentNumber', ''),
                    'driver_name': f"{driver_data.get('givenName', '')} {driver_data.get('familyName', '')}".strip(),
                    'constructor_ids': constructors,
                    'constructor_names': constructor_names,
                    'points': float(standing.get('points', 0.0)),
                    'wins': int(standing.get('wins', 0))
                })
        
        return pd.DataFrame(drivers_standings)
    
    def get_constructor_standings(self, season: Union[int, str], round_num: Optional[Union[int, str]] = None) -> pd.DataFrame:
        """
        Get constructor/team standings for a specific season and optionally after a specific round.
        
        Args:
            season: F1 season (year).
            round_num: Round number or 'current' for the latest round.
            
        Returns:
            DataFrame with constructor standings.
        """
        round_str = f"/{round_num}" if round_num is not None else ""
        url = f"{self.base_url}/{season}{round_str}/constructorStandings.json"
        
        data = self._make_request(url)
        standings_data = data['MRData']['StandingsTable']['StandingsLists']
        
        if not standings_data:
            return pd.DataFrame()
        
        constructor_standings = []
        for standing_list in standings_data:
            season = standing_list['season']
            round_num = standing_list['round']
            
            for standing in standing_list['ConstructorStandings']:
                constructor_data = standing['Constructor']
                
                constructor_standings.append({
                    'season': season,
                    'round': round_num,
                    'position': int(standing['position']),
                    'constructor_id': constructor_data['constructorId'],
                    'constructor_name': constructor_data['name'],
                    'nationality': constructor_data['nationality'],
                    'points': float(standing['points']),
                    'wins': int(standing['wins'])
                })
        
        return pd.DataFrame(constructor_standings)
    
    def get_lap_times(self, season: Union[int, str], round_num: Union[int, str], 
                       driver_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get lap times for a specific race and optionally for a specific driver.
        
        Args:
            season: F1 season (year).
            round_num: Round number.
            driver_id: Optional driver ID to filter results.
            
        Returns:
            DataFrame with lap times.
        """
        driver_str = f"drivers/{driver_id}" if driver_id is not None else ""
        url = f"{self.base_url}/{season}/{round_num}/{driver_str}/laps.json"
        
        data = self._make_request(url)
        races_data = data['MRData']['RaceTable']['Races']
        
        if not races_data:
            return pd.DataFrame()
        
        lap_times = []
        for race_data in races_data:
            race_name = race_data['raceName']
            circuit_name = race_data['Circuit']['circuitName']
            race_date = race_data['date']
            
            for lap in race_data['Laps']:
                lap_number = int(lap['number'])
                
                for timing in lap['Timings']:
                    lap_times.append({
                        'season': season,
                        'round': round_num,
                        'race_name': race_name,
                        'circuit_name': circuit_name,
                        'race_date': race_date,
                        'lap_number': lap_number,
                        'driver_id': timing['driverId'],
                        'position': int(timing['position']),
                        'lap_time': timing['time']
                    })
        
        return pd.DataFrame(lap_times)
    
    def get_circuits(self, season: Optional[Union[int, str]] = None) -> pd.DataFrame:
        """
        Get circuit information, optionally for a specific season.
        
        Args:
            season: Optional F1 season (year).
            
        Returns:
            DataFrame with circuit information.
        """
        season_str = f"/{season}" if season is not None else ""
        url = f"{self.base_url}{season_str}/circuits.json"
        
        data = self._make_request(url)
        circuits_data = data['MRData']['CircuitTable']['Circuits']
        
        if not circuits_data:
            return pd.DataFrame()
        
        circuits = []
        for circuit in circuits_data:
            location = circuit['Location']
            
            circuits.append({
                'circuit_id': circuit['circuitId'],
                'circuit_name': circuit['circuitName'],
                'latitude': float(location['lat']),
                'longitude': float(location['long']),
                'locality': location['locality'],
                'country': location['country']
            })
        
        return pd.DataFrame(circuits) 