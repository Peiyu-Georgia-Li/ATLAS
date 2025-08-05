"""
Fetch Open LLM Leaderboard data directly from the discovered API endpoint.
This script downloads the data and saves it as JSON.
"""

import json
import requests

# API endpoint discovered from our network traffic analysis
LEADERBOARD_API_URL = "https://open-llm-leaderboard-open-llm-leaderboard.hf.space/api/leaderboard/formatted"
OUTPUT_FILE = "open_llm_leaderboard_data.json"

def fetch_leaderboard_data():
    """Fetch leaderboard data directly from the API endpoint."""
    print(f"Fetching data from {LEADERBOARD_API_URL}...")
    
    try:
        response = requests.get(LEADERBOARD_API_URL)
        
        if response.status_code != 200:
            print(f"Error: API request failed with status code {response.status_code}")
            return None
        
        data = response.json()
        print(f"Successfully fetched data: {len(str(data))} characters")
        
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def main():
    """Main function to fetch data and save to JSON file."""
    data = fetch_leaderboard_data()
    
    if not data:
        print("Failed to fetch leaderboard data.")
        return
    
    # Save data to JSON file
    print(f"Saving data to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Data saved to {OUTPUT_FILE}")
    
    # Analyze the structure of the data
    print("\nAnalyzing data structure:")
    if isinstance(data, dict):
        print(f"Data is a dictionary with {len(data)} keys")
        print(f"Keys: {list(data.keys())}")
        
        # Check for nested arrays that might contain model entries
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                print(f"\nKey '{key}' contains a list with {len(value)} items")
                if len(value) > 0 and isinstance(value[0], dict):
                    print(f"First item keys: {list(value[0].keys())[:10]}")
                    print("\nSample entry:")
                    print(json.dumps(value[0], indent=2))
    
    elif isinstance(data, list):
        print(f"Data is a list with {len(data)} items")
        if len(data) > 0:
            print(f"First item keys: {list(data[0].keys())[:10]}")
            print("\nSample entry:")
            print(json.dumps(data[0], indent=2))

if __name__ == "__main__":
    main()
