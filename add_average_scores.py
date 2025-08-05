#!/usr/bin/env python3
"""
Script to extract average_score from open_llm_leaderboard_data.json for models in models_above_threshold.json
"""

import json
import argparse
from tqdm import tqdm

def load_json_file(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Extract average_score for models in threshold JSON.")
    parser.add_argument("--input", default="models_above_threshold.json", help="Input file with models above threshold")
    parser.add_argument("--leaderboard", default="open_llm_leaderboard_data.json", help="Leaderboard data file")
    parser.add_argument("--output", default="models_with_scores.json", help="Output file with added average scores")
    args = parser.parse_args()
    
    # Load models above threshold
    print(f"Loading models from {args.input}...")
    models_data = load_json_file(args.input)
    print(f"Found {len(models_data)} models above threshold.")
    
    # Load leaderboard data
    print(f"Loading leaderboard data from {args.leaderboard}...")
    leaderboard_data = load_json_file(args.leaderboard)
    print(f"Found {len(leaderboard_data)} entries in leaderboard.")
    
    # Create a lookup dictionary for faster access
    model_name_to_data = {}
    print("Building lookup dictionary...")
    for entry in leaderboard_data:
        if 'model' in entry and 'name' in entry['model']:
            model_name = entry['model']['name']
            if 'average_score' in entry['model']:
                model_name_to_data[model_name] = {
                    'average_score': entry['model'].get('average_score'),
                    'architecture': entry['model'].get('architecture', ''),
                    'precision': entry['model'].get('precision', ''),
                    'type': entry['model'].get('type', '')
                }
    
    # Add average_score to each model in the threshold list
    print("Adding average scores to models...")
    models_with_scores = []
    models_not_found = []
    
    for model_entry in tqdm(models_data, desc="Processing models"):
        model_name = model_entry['model']
        new_entry = {
            'model': model_name,
            'task_score': model_entry['score']  # Rename to clarify this is the specific task score
        }
        
        if model_name in model_name_to_data:
            # Add leaderboard data
            leaderboard_info = model_name_to_data[model_name]
            new_entry['average_score'] = leaderboard_info['average_score']
            new_entry['architecture'] = leaderboard_info['architecture']
            new_entry['precision'] = leaderboard_info['precision']
            new_entry['model_type'] = leaderboard_info['type']
        else:
            # Mark as not found in leaderboard
            new_entry['average_score'] = None
            models_not_found.append(model_name)
        
        models_with_scores.append(new_entry)
    
    # Sort by task_score (highest first)
    models_with_scores.sort(key=lambda x: x['task_score'], reverse=True)
    
    # Save the combined data
    with open(args.output, 'w') as f:
        json.dump(models_with_scores, f, indent=2)
    
    print(f"Saved {len(models_with_scores)} models to {args.output}")
    print(f"Models not found in leaderboard: {len(models_not_found)}")
    
    if models_not_found:
        print("First 5 models not found:")
        for i, model in enumerate(models_not_found[:5]):
            print(f"{i+1}. {model}")

if __name__ == "__main__":
    main()
