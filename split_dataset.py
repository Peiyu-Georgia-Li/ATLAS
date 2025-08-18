#!/usr/bin/env python3
"""
Script to shuffle and randomly split cleaned_8b_model.csv into 3 datasets.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse

def split_dataset(input_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Shuffle and split a CSV file into train, validation, and test datasets.
    
    Args:
        input_file (str): Path to the input CSV file
        train_ratio (float): Proportion for training set
        val_ratio (float): Proportion for validation set
        test_ratio (float): Proportion for test set
        random_seed (int): Random seed for reproducibility
    
    Returns:
        tuple: Paths to the created files (train_path, val_path, test_path)
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    # Read the CSV file
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    # Get the total number of rows
    total_rows = len(df)
    print(f"Total rows: {total_rows}")
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Calculate split sizes
    train_size = int(train_ratio * total_rows)
    val_size = int(val_ratio * total_rows)
    
    # Split into train and temporary set
    train_df = df.iloc[:train_size]
    temp_df = df.iloc[train_size:]
    
    # Split temporary set into validation and test
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(temp_df, test_size=(1-val_test_ratio), random_state=random_seed)
    
    # Create output file paths
    base_name = os.path.splitext(input_file)[0]
    train_path = f"{base_name}_train.csv"
    val_path = f"{base_name}_val.csv"
    test_path = f"{base_name}_test.csv"
    
    # Save the datasets
    print(f"Saving training set ({len(train_df)} rows) to {train_path}")
    train_df.to_csv(train_path, index=False)
    
    print(f"Saving validation set ({len(val_df)} rows) to {val_path}")
    val_df.to_csv(val_path, index=False)
    
    print(f"Saving test set ({len(test_df)} rows) to {test_path}")
    test_df.to_csv(test_path, index=False)
    
    return train_path, val_path, test_path

def main():
    parser = argparse.ArgumentParser(description='Split a CSV file into train, validation, and test sets')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('--train-ratio', type=float, default=0.3333, help='Proportion for training set (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.3333, help='Proportion for validation set (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.3334, help='Proportion for test set (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Check if the ratios sum to 1
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-10:
        print(f"Warning: Ratios sum to {total_ratio}, not 1.0. Normalizing...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
    
    # Check if the input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return 1
    
    # Split the dataset
    try:
        train_path, val_path, test_path = split_dataset(
            args.input_file, 
            args.train_ratio, 
            args.val_ratio, 
            args.test_ratio,
            args.seed
        )
        print("\nDataset split successfully!")
        print(f"Train: {train_path}")
        print(f"Validation: {val_path}")
        print(f"Test: {test_path}")
        return 0
    except Exception as e:
        print(f"Error splitting dataset: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
