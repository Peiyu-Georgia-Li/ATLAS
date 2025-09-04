#!/usr/bin/env python3
import pandas as pd
import re
import argparse

def filter_8b_models(input_file, output_file, print_removed=False):
    """
    Filter the CSV file to keep only rows where the model name contains -8b, -8B, _8b, or _8B
    (effectively removing rows containing 78b models)
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        print_removed: Whether to print the removed rows
    """
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Get the name of the first column (usually 'Unnamed: 0')
    model_col = df.columns[0]
    
    # Count rows before filtering
    total_rows = len(df)
    
    # Filter to keep only rows with -8b, -8B, _8b, or _8B in the model name
    # This effectively removes models with 78b
    pattern = r'[-_x]8[bB]'
    filtered_df = df[df[model_col].str.contains(pattern, regex=True)]
    
    # Get the removed rows
    removed_df = df[~df[model_col].str.contains(pattern, regex=True)]
    
    # Count rows after filtering
    kept_rows = len(filtered_df)
    removed_rows = total_rows - kept_rows
    
    # Save filtered dataframe to output file
    filtered_df.to_csv(output_file, index=False)
    
    print(f"Filtering complete:")
    print(f"  - Original rows: {total_rows}")
    print(f"  - Kept rows: {kept_rows}")
    print(f"  - Removed rows: {removed_rows}")
    print(f"Output saved to: {output_file}")
    
    # Print removed rows if requested
    if print_removed and removed_rows > 0:
        print("\nRemoved rows (model names):")
        for model_name in removed_df[model_col].values:
            print(f"  - {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter CSV to keep only 8B models')
    parser.add_argument('--input_file', default='8b_leaderboard_mmlu_pro_response_matrix_math.csv',
                        help='Input CSV file path (default: 8b_leaderboard_mmlu_pro_response_matrix_math.csv)')
    parser.add_argument('--output_file', default='filtered_8b_leaderboard_mmlu_pro_response_matrix_math.csv',
                        help='Output CSV file path (default: filtered_8b_leaderboard_mmlu_pro_response_matrix_math.csv)')
    parser.add_argument('--print_removed', action='store_true',
                        help='Print the names of removed models')
    
    args = parser.parse_args()
    
    filter_8b_models(args.input_file, args.output_file, args.print_removed)
