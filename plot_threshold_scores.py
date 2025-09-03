#!/usr/bin/env python3
"""
Script to plot the distribution of average scores for models in models_above_threshold.json
and compare with a Gaussian distribution
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

def load_json_file(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)

    
    

def extract_threshold_model_scores(threshold_models, leaderboard_data):
    """Extract average scores for models in threshold list."""
    # Create set of model names from threshold models for faster lookup
    threshold_model_names = {model_data['model'] for model_data in threshold_models}
    
    # Create a lookup dictionary from leaderboard data
    model_to_avg_score = {}
    for entry in tqdm(leaderboard_data, desc="Building score lookup"):
        if 'model' in entry and 'name' in entry['model'] and 'average_score' in entry['model']:
            model_name = entry['model']['name']
            avg_score = entry['model']['average_score']
            if avg_score is not None and isinstance(avg_score, (int, float)):
                model_to_avg_score[model_name] = avg_score
    
    # Extract scores for threshold models
    scores = []
    models_found = []
    models_missing = []
    
    for model_data in tqdm(threshold_models, desc="Extracting scores"):
        model_name = model_data['model']
        if model_name in model_to_avg_score:
            scores.append(model_to_avg_score[model_name])
            models_found.append(model_name)
        else:
            models_missing.append(model_name)
    
    print(f"Found average scores for {len(models_found)} out of {len(threshold_models)} models.")
    if models_missing:
        print(f"Missing average scores for {len(models_missing)} models.")
        print(f"First 5 missing models: {models_missing[:5]}")
    
    return np.array(scores)

def plot_score_distribution(scores, output_file=None):
    """Plot histogram of scores and compare with Gaussian."""
    plt.figure(figsize=(12, 8))
    
    # Plot histogram with density=True to get probability density
    n, bins, patches = plt.hist(scores, bins=30, density=True, alpha=0.7, 
                               label='Average Score Distribution')
    
    # Fit a normal distribution to the data
    mu, sigma = stats.norm.fit(scores)
    
    # Calculate skewness and kurtosis
    skewness = stats.skew(scores)
    kurtosis = stats.kurtosis(scores)
    
    # Plot the PDF (probability density function)
    x = np.linspace(min(scores), max(scores), 100)
    pdf = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, pdf, 'r-', linewidth=2, label=f'Fitted Gaussian: μ={mu:.2f}, σ={sigma:.2f}')
    
    # Calculate goodness of fit with Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.kstest(scores, 'norm', args=(mu, sigma))
    
    # Add title and labels
    plt.title(f'Distribution of Average Scores for 8B Models\nKS test: statistic={ks_statistic:.4f}, p-value={p_value:.4f}', 
             fontsize=14)
    plt.xlabel('Average Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add text box with statistics
    stats_text = f'Number of models: {len(scores)}\n'
    stats_text += f'Mean: {np.mean(scores):.4f}\n'
    stats_text += f'Median: {np.median(scores):.4f}\n'
    stats_text += f'Std Dev: {np.std(scores):.4f}\n'
    stats_text += f'Min: {np.min(scores):.4f}\n'
    stats_text += f'Max: {np.max(scores):.4f}'
    
    plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add skewness and kurtosis information
    skew_kurt_text = f'Skewness: {skewness:.4f}\n'
    skew_kurt_text += f'Kurtosis: {kurtosis:.4f}\n'
    skew_kurt_text += f'(For a perfect Gaussian: Skewness=0, Kurtosis=0)'
    
    plt.annotate(skew_kurt_text, xy=(0.98, 0.98), xycoords='axes fraction',
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Add interpretation of Gaussian fit
    if p_value < 0.05:
        fit_text = "The distribution differs significantly from a Gaussian (p<0.05)"
    else:
        fit_text = "The distribution is consistent with a Gaussian (p>=0.05)"
    
    plt.annotate(fit_text, xy=(0.98, 0.02), xycoords='axes fraction',
                fontsize=10, horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot distribution of average scores for threshold models and compare with Gaussian.")
    parser.add_argument("--threshold", default="models_above_threshold.json", help="Models above threshold file")
    parser.add_argument("--leaderboard", default="open_llm_leaderboard_data.json", help="Leaderboard data file")
    parser.add_argument("--output", default="threshold_scores_distribution.png", help="Output image file")
    args = parser.parse_args()
    
    # Load threshold models
    print(f"Loading threshold models from {args.threshold}...")
    threshold_models = load_json_file(args.threshold)
    print(f"Found {len(threshold_models)} models above threshold.")
    
    # Load leaderboard data
    print(f"Loading leaderboard data from {args.leaderboard}...")
    leaderboard_data = load_json_file(args.leaderboard)
    print(f"Found {len(leaderboard_data)} entries in leaderboard.")
    
    # Extract scores for threshold models
    scores = extract_threshold_model_scores(threshold_models, leaderboard_data)
    print(f"Extracted {len(scores)} valid scores for threshold models.")
    
    if len(scores) == 0:
        print("No valid scores found. Cannot generate plot.")
        return
    
    # Plot the distribution
    print("Generating plot...")
    plot_score_distribution(scores, args.output)
    
    # Additional analysis: skewness and kurtosis
    skewness = stats.skew(scores)
    kurtosis = stats.kurtosis(scores)
    print(f"Distribution Skewness: {skewness:.4f}")
    print(f"Distribution Kurtosis: {kurtosis:.4f}")
    print("(For a perfect Gaussian: Skewness=0, Kurtosis=0)")


def calculate_average_score(csv_file_path):
    """
    Calculate the average score for each model in the CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file containing model scores
        
    Returns:
        dict: Dictionary with model names as keys and their average scores as values
    """
    import csv
    import os
    
    # Check if file exists
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    # Calculate average score for each model
    results = {}
    
    # Read the CSV file
    with open(csv_file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        
        # Read header row
        header = next(csv_reader)
        
        # Process each row (model)
        for row in csv_reader:
            if not row:  # Skip empty rows
                continue
                
            model_name = row[0]  # First column is the model name
            
            # Calculate average by taking mean of all score columns (columns 1 to end)
            scores = [int(score) for score in row[1:] if score.strip()]  # Convert to integers
            
            if scores:  # Check if there are any scores
                average_score = sum(scores) / len(scores)
                # Store result
                results[model_name] = average_score
    
    return results

def plot():
    """
    Plot the distribution of average scores from the cleaned_8b_model.csv file
    """
    parser = argparse.ArgumentParser(description="Plot distribution of average scores for 8B models and compare with Gaussian.")
    parser.add_argument("--csv", default="/Users/lipeiyu/llmbenchmark/cleaned_8b_model_train.csv", help="Path to CSV file with model scores")
    parser.add_argument("--output", default="8b_scores_distribution_train.png", help="Output image file")
    args = parser.parse_args()
    
    # Calculate average scores from CSV
    print(f"Calculating average scores from {args.csv}...")
    average_scores = calculate_average_score(args.csv)
    
    # Convert dictionary values to a numpy array for plotting
    scores_array = np.array(list(average_scores.values()))
    
    print(f"Found {len(scores_array)} models with valid scores.")
    print(f"Score range: {np.min(scores_array):.4f} to {np.max(scores_array):.4f}")
    print(f"Mean score: {np.mean(scores_array):.4f}")
    
    # Plot the distribution
    print("Generating plot...")
    plot_score_distribution(scores_array, args.output)
    
    # Additional analysis: skewness and kurtosis
    skewness = stats.skew(scores_array)
    kurtosis = stats.kurtosis(scores_array)
    print(f"Distribution Skewness: {skewness:.4f}")
    print(f"Distribution Kurtosis: {kurtosis:.4f}")
    print("(For a perfect Gaussian: Skewness=0, Kurtosis=0)")
    
if __name__ == "__main__":
    plot()
