#!/usr/bin/env python3
import pandas as pd
import os
import argparse

def filter_8b_models(csv_path, output_path=None):
    """
    Filter rows in a CSV file where the model name contains '8b' or '8B'.
    
    Args:
        csv_path (str): Path to the input CSV file
        output_path (str, optional): Path to save the filtered CSV. If None, returns the DataFrame
        
    Returns:
        pd.DataFrame: Filtered DataFrame if output_path is None, otherwise None
    """
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")
    
    # Check if 'model' column exists
    if 'model' not in df.columns:
        raise ValueError("CSV file does not contain a 'model' column")
    
    # Filter rows where model name contains '8b' or '8B'
    filtered_df = df[df['model'].str.contains('8[bB]', regex=True)]
    
    # Save to file if output_path is provided
    if output_path:
        filtered_df.to_csv(output_path, index=False)
        print(f"Filtered data saved to {output_path}")
        return None
    
    # Otherwise return the filtered DataFrame
    return filtered_df

def main():
    parser = argparse.ArgumentParser(description='Filter models containing 8b/8B in their names')
    parser.add_argument('csv_path', help='Path to the input CSV file')
    parser.add_argument('--output', '-o', help='Path to save the filtered CSV (optional)')
    args = parser.parse_args()
    
    filter_8b_models(args.csv_path, args.output)

if __name__ == "__main__":
    main()
