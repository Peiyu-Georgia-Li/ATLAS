#!/usr/bin/env python3
"""
Create a CSV response matrix for IRT analysis of MMLU data.
Rows: Models from models_above_threshold.json
Columns: Document IDs from MMLU Pro dataset
Values: Binary accuracy scores (0/1) for each model-item pair
"""

import json
import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import argparse
import numpy as np

BENCHMARK_NAME = "leaderboard_bbh_tracking_shuffled_objects_five_objects"
def load_json_file(filename):
    """Load a JSON file and return the data."""
    with open(filename, 'r') as f:
        return json.load(f)

def load_model_responses(model_name):
    """
    Load item-level responses for a specific model from the dataset.
    
    Args:
        model_name: Name of the model
        dataset_path: Path to the dataset
        
    Returns:
        DataFrame with item-level responses and accuracy
    """
    dataset_path = f"open-llm-leaderboard/{model_name.replace('/', '__')}-details"
    
    try: 
        # Load the dataset
        data = load_dataset(
            dataset_path,
            name=f"{model_name.replace('/', '__')}__{BENCHMARK_NAME}",
            split="latest"
        )
    except (ValueError, FileNotFoundError, Exception) as e:
        if "DataFilesNotFoundError" in str(type(e)):
            print(f"Dataset path not found for {model_name}. Skipping.")
        else:
            print(f"Could not load data for {model_name}: {str(e)}")
        return pd.DataFrame()
    
    # Convert to DataFrame for easier processing
    items_data = []
    for item in data:
        # Extract item ID (document ID)
        doc_id = item.get('doc_id', None)
        
        # Extract accuracy (0 or 1)
        acc = item.get('acc_norm', item.get('acc', None))
        
        if doc_id is not None and acc is not None:
            items_data.append({
                'doc_id': doc_id,
                'acc_norm': acc
            })
            
    return pd.DataFrame(items_data)
    


def create_response_matrix(models, output_file, dataset_path="open-llm-leaderboard"):
    """
    Create a response matrix CSV file for IRT analysis.
    
    Args:
        models: List of model names
        output_file: Path to save the output CSV
        dataset_path: Path to the dataset
    """
    # Dictionary to store responses for each model
    all_responses = {}
    all_item_ids = set()
    
    # Load responses for each model
    for model_name in tqdm(models, desc="Loading model responses"):
        responses_df = load_model_responses(model_name)
        if responses_df is not None and not responses_df.empty:

            # Store responses in dictionary
            model_responses = dict(zip(responses_df['doc_id'], responses_df['acc_norm']))
            all_responses[model_name] = model_responses
            
            # Update set of all item IDs
            all_item_ids.update(responses_df['doc_id'])
    
    # Convert all_item_ids to a sorted list
    all_item_ids = sorted(list(all_item_ids))
    
    # Create DataFrame for response matrix
    response_matrix = pd.DataFrame(index=all_responses.keys(), columns=all_item_ids)
    
    # Fill in the response matrix
    for model_name, responses in all_responses.items():
        for item_id in all_item_ids:
            response_matrix.loc[model_name, item_id] = responses.get(item_id, np.nan)
    
    # Save the response matrix to CSV
    response_matrix.to_csv(output_file)
    print(f"Response matrix saved to {output_file}")
    
    # Print some statistics
    print(f"Response matrix shape: {response_matrix.shape}")
    print(f"Number of models: {len(response_matrix.index)}")
    print(f"Number of items: {len(response_matrix.columns)}")
    
    # Calculate average accuracy per model
    model_accuracies = response_matrix.mean(axis=1)
    print(f"Average model accuracy: {model_accuracies.mean():.4f}")
    print(f"Min model accuracy: {model_accuracies.min():.4f}")
    print(f"Max model accuracy: {model_accuracies.max():.4f}")
    
    # Calculate percentage of items answered by all models
    items_completion = response_matrix.notna().mean(axis=0)
    print(f"Average item completion rate: {items_completion.mean():.4f}")
    
    return response_matrix

def main():
    parser = argparse.ArgumentParser(description="Create a response matrix for IRT analysis")
    parser.add_argument("--threshold", default="models_above_threshold.json", help="Models above threshold file")
    parser.add_argument("--output", default=f"{BENCHMARK_NAME}_response_matrix.csv", help="Output CSV file")
    parser.add_argument("--dataset", default="open-llm-leaderboard", help="Dataset path")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of models to process (for testing)")
    args = parser.parse_args()
    
    # Load threshold models
    print(f"Loading threshold models from {args.threshold}...")
    threshold_models = load_json_file(args.threshold)
    print(f"Found {len(threshold_models)} models above threshold.")
    
    # Check the structure of the first model to determine how to extract model names
    if len(threshold_models) > 0:
        first_model = threshold_models[0]
        print(f"Sample model structure: {first_model}")
        
        # Extract model names based on the structure
        if isinstance(first_model, str):
            # Models are directly stored as strings
            model_names = threshold_models
        elif isinstance(first_model, dict):
            # Models are dictionaries, determine which key to use
            if "name" in first_model:
                model_names = [model["name"] for model in threshold_models]
            elif "id" in first_model:
                model_names = [model["id"] for model in threshold_models]
            elif "model" in first_model:
                model_names = [model["model"] for model in threshold_models]
            else:
                # If we can't determine the key, use the first key in the dictionary
                key = list(first_model.keys())[0]
                model_names = [model.get(key, str(i)) for i, model in enumerate(threshold_models)]
                print(f"Using key '{key}' to extract model names")
        else:
            raise ValueError(f"Unexpected model format in threshold file: {type(first_model)}")
    else:
        model_names = []
    
    # Limit the number of models for testing if specified
    if args.limit > 0 and args.limit < len(model_names):
        print(f"Limiting to first {args.limit} models for testing")
        model_names = model_names[:args.limit]
    
    # Create response matrix
    response_matrix = create_response_matrix(model_names, args.output, args.dataset)

if __name__ == "__main__":
    main()
