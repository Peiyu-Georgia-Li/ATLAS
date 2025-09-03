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
import shutil
from datasets import config

BENCHMARK_NAME = "leaderboard_mmlu_pro"
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

        # Extract category ⚠️!!! just for mmlu-prop !!!
        category = item.get('doc', None)
        if category is not None:
            category = category.get('category', 'unknown')
        else:
            category = 'unknown'
        
        # Extract accuracy (0 or 1)
        acc = item.get('acc_norm', item.get('acc', None))
        
        if doc_id is not None and acc is not None:
            items_data.append({
                'doc_id': doc_id,
                'category': category,
                'acc_norm': acc
            })
    
    result_df = pd.DataFrame(items_data)
    # Delete the Hugging Face dataset cache for this specific dataset
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    cache_dir2 = os.path.expanduser("~/.cache/huggingface/hub")
    shutil.rmtree(cache_dir, ignore_errors=True)
    shutil.rmtree(cache_dir2, ignore_errors=True)
    print(f"Deleting cache for {model_name}...")

    
    return result_df
    


def create_response_matrix(models, output_file):
    """
    Create response matrix CSV files for IRT analysis, grouped by category.
    
    Args:
        models: List of model names
        output_file: Base path to save the output CSVs
    """
    # Dictionary to store responses for each model
    all_responses = {}
    all_item_ids = set()
    item_categories = {}  # Dictionary to store category for each item
    
    # Load responses for each model
    for model_name in tqdm(models, desc="Loading model responses"):
        responses_df = load_model_responses(model_name)
        if responses_df is not None and not responses_df.empty:
            # Store responses in dictionary
            model_responses = dict(zip(responses_df['doc_id'], responses_df['acc_norm']))
            all_responses[model_name] = model_responses
            
            # Update set of all item IDs
            all_item_ids.update(responses_df['doc_id'])
            
            # Store category information for each item
            for _, row in responses_df.iterrows():
                item_categories[row['doc_id']] = row['category']
    
    # Convert all_item_ids to a sorted list
    all_item_ids = sorted(list(all_item_ids))
    
    # Group items by category
    category_items = {}
    for item_id in all_item_ids:
        category = item_categories.get(item_id, 'unknown')
        if category not in category_items:
            category_items[category] = []
        category_items[category].append(item_id)
    
    # Create and save response matrices for each category
    all_matrices = {}
    
    # First create a combined matrix for all items
    response_matrix_all = pd.DataFrame(index=all_responses.keys(), columns=all_item_ids)
    for model_name, responses in all_responses.items():
        for item_id in all_item_ids:
            response_matrix_all.loc[model_name, item_id] = responses.get(item_id, np.nan)
    
    # Save the combined response matrix
    response_matrix_all.to_csv(output_file)
    print(f"Combined response matrix saved to {output_file}")
    all_matrices['all'] = response_matrix_all
    
    # Create and save matrices for each category
    for category, items in category_items.items():
        # Skip if category is None or empty
        if not category or category == 'None':
            category = 'unknown'
            
        # Create output filename for this category
        category_output = output_file.replace('.csv', f'_{category}.csv')
        
        # Create DataFrame for this category's response matrix
        response_matrix = pd.DataFrame(index=all_responses.keys(), columns=items)
        
        # Fill in the response matrix
        for model_name, responses in all_responses.items():
            for item_id in items:
                response_matrix.loc[model_name, item_id] = responses.get(item_id, np.nan)
        
        # Save the response matrix to CSV
        response_matrix.to_csv(category_output)
        print(f"Response matrix for category '{category}' saved to {category_output}")
        all_matrices[category] = response_matrix
    
    # Print statistics for the combined matrix
    print(f"\nCombined response matrix statistics:")
    print(f"Response matrix shape: {response_matrix_all.shape}")
    print(f"Number of models: {len(response_matrix_all.index)}")
    print(f"Number of items: {len(response_matrix_all.columns)}")
    
    # Calculate average accuracy per model for combined matrix
    model_accuracies = response_matrix_all.mean(axis=1)
    print(f"Average model accuracy: {model_accuracies.mean():.4f}")
    print(f"Min model accuracy: {model_accuracies.min():.4f}")
    print(f"Max model accuracy: {model_accuracies.max():.4f}")
    
    # Calculate percentage of items answered by all models
    items_completion = response_matrix_all.notna().mean(axis=0)
    print(f"Average item completion rate: {items_completion.mean():.4f}")
    
    # Print category statistics
    print(f"\nCategory statistics:")
    for category, matrix in all_matrices.items():
        if category != 'all':
            print(f"Category '{category}': {len(matrix.columns)} items")
    
    return all_matrices

def main():
    parser = argparse.ArgumentParser(description="Create response matrices for IRT analysis, grouped by category")
    parser.add_argument("--threshold", default="data/models_above_threshold.json", help="Models above threshold file")
    parser.add_argument("--output", default=f"{BENCHMARK_NAME}_response_matrix.csv", help="Base name for output CSV files")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of models to process (for testing)")
    parser.add_argument("--keep-cache", action="store_true", help="Keep Hugging Face cache after execution")
    parser.add_argument("--output-dir", default="mmlu_pro", help="Directory to save category-specific output files (default: same as output)")
    args = parser.parse_args()
    
    # Set up output directory if specified
    output_file = args.output
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_basename = os.path.basename(args.output)
        output_file = os.path.join(args.output_dir, output_basename)
    
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
    
    # Create response matrices grouped by category
    response_matrices = create_response_matrix(model_names, output_file)
    
    print(f"\nCreated {len(response_matrices)} response matrices (1 combined + {len(response_matrices)-1} category-specific)")
    print(f"Files saved with base name: {output_file}")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
        
    # Return the combined matrix for backward compatibility
    return response_matrices.get('all', None)



if __name__ == "__main__":
    main()
