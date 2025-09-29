#!/usr/bin/env python3
"""
Script to split the gaussian_sampled_truthfulqa_response_matrix.csv into train and test sets.
The split is done with stratification based on average scores, with 90% train and 10% test.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
sys.path.append('..')
from plot_model_scores import plot_score_distribution
from scipy import stats  # Added for the skewness and kurtosis calculations

# BENCHMARK_NAME = "truthfulqa"
# BENCHMARK_NAME = "mmlu"
# BENCHMARK_NAME = "winogrande"
BENCHMARK_NAME="hellaswag"


def main():
    # Load the data
    print("Loading data...")
    output_prefix = "gaussian_sampled_"+BENCHMARK_NAME #⚠️
    file_path = "gaussian_sampled_"+BENCHMARK_NAME+"_response_matrix.csv" #⚠️
    data = pd.read_csv(file_path, index_col=0)
    
    # Calculate average score for each model (row)
    print("Calculating average scores...")
    data['avg_score'] = data.mean(axis=1)
    
    # Create bins for stratification based on average score
    # This helps ensure both train and test sets have similar distributions of model performance
    n_bins = 10  # Number of bins for stratification
    data['score_bin'] = pd.qcut(data['avg_score'], n_bins, labels=False, duplicates='drop')
    
    # Print score distribution
    print(f"Score distribution in {n_bins} bins:")
    bin_counts = data['score_bin'].value_counts().sort_index()
    for bin_idx, count in bin_counts.items():
        bin_min = data[data['score_bin'] == bin_idx]['avg_score'].min()
        bin_max = data[data['score_bin'] == bin_idx]['avg_score'].max()
        print(f"Bin {bin_idx}: {count} models, score range: {bin_min:.4f}-{bin_max:.4f}")
    
    # Split into features (X) and stratification column (y)
    X = data.drop(['avg_score', 'score_bin'], axis=1)
    y = data['score_bin']
    
    # Perform the stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    
    # Verify the split
    print(f"Train set size: {len(X_train)} models")
    print(f"Test set size: {len(X_test)} models")
    
    # Verify stratification worked
    train_bin_counts = y_train.value_counts().sort_index()
    test_bin_counts = y_test.value_counts().sort_index()
    
    print("\nStratification check:")
    for bin_idx in bin_counts.index:
        if bin_idx in train_bin_counts.index and bin_idx in test_bin_counts.index:
            train_pct = train_bin_counts[bin_idx] / len(y_train) * 100
            test_pct = test_bin_counts[bin_idx] / len(y_test) * 100
            print(f"Bin {bin_idx}: Train {train_pct:.1f}%, Test {test_pct:.1f}%")
    
    # Save the split datasets to new files
    train_file = file_path.replace('.csv', '_train.csv')
    test_file = file_path.replace('.csv', '_test.csv')
    
    X_train.to_csv(train_file)
    X_test.to_csv(test_file)
    
    print(f"Saved train set to: {train_file}")
    print(f"Saved test set to: {test_file}")
    
    # Additionally, save files with average scores
    train_with_scores = pd.concat([X_train, data.loc[X_train.index, ['avg_score']]], axis=1)
    test_with_scores = pd.concat([X_test, data.loc[X_test.index, ['avg_score']]], axis=1)
    
    train_file_with_scores = file_path.replace('.csv', '_train_with_scores.csv')
    test_file_with_scores = file_path.replace('.csv', '_test_with_scores.csv')
    
    train_with_scores.to_csv(train_file_with_scores)
    test_with_scores.to_csv(test_file_with_scores)
    
    print(f"Saved train set with scores to: {train_file_with_scores}")
    print(f"Saved test set with scores to: {test_file_with_scores}")
    
    # Plot the score distributions
    plot_scores(train_with_scores, test_with_scores, output_prefix)



def plot_scores(train_with_scores, test_with_scores, output_prefix="gaussian_sampled_truthfulqa"):
    # Convert dataframe values to a numpy array for plotting
    train_scores_array = np.array(train_with_scores['avg_score'].values)
    test_scores_array = np.array(test_with_scores['avg_score'].values)
    
    print("\nAnalyzing training set distribution:")
    print(f"Found {len(train_scores_array)} models with valid scores.")
    print(f"Score range: {np.min(train_scores_array):.4f} to {np.max(train_scores_array):.4f}")
    print(f"Mean score: {np.mean(train_scores_array):.4f}")
    
    # Plot the distribution
    print("Generating train set plot...")
    plot_score_distribution(train_scores_array, output_prefix + "_train.png")
    
    # Additional analysis: skewness and kurtosis
    skewness_train = stats.skew(train_scores_array)
    kurtosis_train = stats.kurtosis(train_scores_array)
    print(f"Distribution Skewness: {skewness_train:.4f}")
    print(f"Distribution Kurtosis: {kurtosis_train:.4f}")
    print("(For a perfect Gaussian: Skewness=0, Kurtosis=0)")

    print("\nAnalyzing test set distribution:")
    print(f"Found {len(test_scores_array)} models with valid scores.")
    print(f"Score range: {np.min(test_scores_array):.4f} to {np.max(test_scores_array):.4f}")
    print(f"Mean score: {np.mean(test_scores_array):.4f}")
    
    # Plot the distribution
    print("Generating test set plot...")
    plot_score_distribution(test_scores_array, output_prefix + "_test.png")
    
    # Additional analysis: skewness and kurtosis
    skewness_test = stats.skew(test_scores_array)
    kurtosis_test = stats.kurtosis(test_scores_array)
    print(f"Distribution Skewness: {skewness_test:.4f}")
    print(f"Distribution Kurtosis: {kurtosis_test:.4f}")
    print("(For a perfect Gaussian: Skewness=0, Kurtosis=0)")

if __name__ == "__main__":
    main()
