# test_jolpica_client.py
import pandas as pd
from jolpica_client import JolpicaClient

def test_client():
    # Initialize the client with default configuration
    config = {
        'cache_ttl': 3600,  # 1 hour cache
    }
    client = JolpicaClient(config)
    
    print("Testing JolpicaClient methods...\n")
    
    # 1. Test race results
    print("1. TESTING RACE RESULTS")
    race_results = client.get_race_results(season=2023, race=1)
    print(f"Got {len(race_results)} race results")
    if not race_results.empty:
        print(race_results[['driver_name', 'team_name', 'position']].head())
    print("\n" + "-"*50 + "\n")
    
    # 2. Test qualifying results
    print("2. TESTING QUALIFYING RESULTS")
    quali_results = client.get_qualifying_results(season=2023, race=1)
    print(f"Got {len(quali_results)} qualifying results")
    if not quali_results.empty:
        print(quali_results[['driver_name', 'team_name', 'position']].head())
    print("\n" + "-"*50 + "\n")
    
    # 3. Test driver standings
    print("3. TESTING DRIVER STANDINGS")
    driver_standings = client.get_driver_standings(season=2023, round_num=5)
    print(f"Got {len(driver_standings)} driver standings")
    if not driver_standings.empty:
        print(driver_standings[['driver_name', 'position', 'points']].head())
    print("\n" + "-"*50 + "\n")
    
    # 4. Test constructor standings
    print("4. TESTING CONSTRUCTOR STANDINGS")
    constructor_standings = client.get_constructor_standings(season=2023, round_num=5)
    print(f"Got {len(constructor_standings)} constructor standings")
    if not constructor_standings.empty:
        print(constructor_standings[['constructor_name', 'position', 'points']].head())
    print("\n" + "-"*50 + "\n")
    
    # 5. Test lap times
    print("5. TESTING LAP TIMES")
    lap_times = client.get_lap_times(season=2023, round_num=1, driver_id='max_verstappen')
    print(f"Got {len(lap_times)} lap times")
    if not lap_times.empty:
        print(lap_times[['lap_number', 'position', 'lap_time']].head())
    print("\n" + "-"*50 + "\n")
    
    # 6. Test circuits
    print("6. TESTING CIRCUITS")
    circuits = client.get_circuits(season=2023)
    print(f"Got {len(circuits)} circuits")
    if not circuits.empty:
        print(circuits[['circuit_name', 'country']].head())
    print("\n" + "-"*50 + "\n")
    
    print("All tests completed.")

if __name__ == "__main__":
    test_client()