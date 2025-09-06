#!/usr/bin/env python3
"""
Script to sample models from filtered_8b_leaderboard_mmlu_pro_response_matrix_math.csv
such that their average scores fit a Gaussian distribution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import random
from scipy import stats


def sample_models_for_gaussian(response_matrix_file, avg_scores_file, output_file, 
                              target_mean=None, target_std=None, num_models=None, plot=True,
                              model_col='Model', score_col='Average Score'):
    """
    Sample models from the response matrix such that their average scores 
    fit a Gaussian distribution centered at the specified mean.
    
    Args:
        response_matrix_file: Path to the CSV file containing model responses (0s and 1s)
        avg_scores_file: Path to the CSV file containing average scores
        output_file: Path to save the sampled models' response matrix
        mean: Mean of the target Gaussian distribution
        std_dev: Standard deviation of the target Gaussian distribution
        num_models: Number of models to sample
        plot: Whether to generate a plot of the distributions
    """
    # Load the response matrix and average scores
    df_responses = pd.read_csv(response_matrix_file)
    df_avg = pd.read_csv(avg_scores_file)
    
    # Ensure the model names match between the two files
    model_names = df_responses['Unnamed: 0'].values
    avg_scores = {}
    
    # Check if the provided column names exist in the dataframe
    if model_col not in df_avg.columns or score_col not in df_avg.columns:
        print(f"Warning: Could not find columns '{model_col}' and/or '{score_col}'")
        print(f"Available columns: {df_avg.columns.tolist()}")
        
        # Try to infer columns - first column is usually model name, second is score
        if len(df_avg.columns) >= 2:
            model_col = df_avg.columns[0]
            score_col = df_avg.columns[1]
            print(f"Using columns '{model_col}' and '{score_col}' instead")
        else:
            raise ValueError("Could not determine model and score columns")
    
    # Create a dictionary mapping model names to their average scores
    for _, row in df_avg.iterrows():
        model_name = row[model_col]
        avg_score = float(row[score_col])
        avg_scores[model_name] = avg_score
    
    # Filter to only include models that are in both files
    valid_models = []
    valid_scores = []
    
    for model in model_names:
        if model in avg_scores:
            valid_models.append(model)
            valid_scores.append(avg_scores[model])
    
    print(f"Found {len(valid_models)} valid models with scores")
    
    # If parameters are not provided, determine them from the data using MLE fitting
    if target_mean is None or target_std is None:
        # Fit Gaussian distribution using Maximum Likelihood Estimation
        target_mean, target_std = stats.norm.fit(valid_scores)
        print(f"Fitted Gaussian parameters: mean={target_mean:.3f}, std_dev={target_std:.3f}")
    else:
        print(f"Using provided parameters: mean={target_mean:.3f}, std_dev={target_std:.3f}")
        
    if num_models is None:
        # Use all available models by default
        num_models = len(valid_models)
        print(f"Using all {num_models} available models")
    else:
        print(f"Using {num_models} models as specified")
    
    # Generate target scores from a Gaussian distribution
    target_scores = np.random.normal(target_mean, target_std, num_models)
    target_scores = np.clip(target_scores, 0, 1)  # Clip to valid score range
    
    # Sort the target scores and the valid scores
    target_scores = np.sort(target_scores)
    valid_models_scores = list(zip(valid_models, valid_scores))
    valid_models_scores.sort(key=lambda x: x[1])
    
    # Sample models that best match the target distribution
    sampled_models = []
    
    # For each target score, find the closest model score
    for target in target_scores:
        closest_idx = min(range(len(valid_models_scores)), 
                          key=lambda i: abs(valid_models_scores[i][1] - target))
        
        # Add the model to our sampled list
        sampled_models.append(valid_models_scores[closest_idx][0])
        
        # Remove the selected model to avoid duplicates
        valid_models_scores.pop(closest_idx)
        
        # If we run out of models, break
        if not valid_models_scores:
            break
    
    # Get the actual scores of the sampled models
    sampled_scores = [avg_scores[model] for model in sampled_models]
    
    # Create a new dataframe with only the sampled models
    df_sampled = df_responses[df_responses['Unnamed: 0'].isin(sampled_models)]
    
    # Save to CSV
    df_sampled.to_csv(output_file, index=False)
    
    print(f"Sampled {len(sampled_models)} models and saved to {output_file}")
    
    # Generate a plot if requested
    if plot:
        plot_file = Path(output_file).with_suffix('.png')
        plot_distributions(sampled_scores, target_mean, target_std, plot_file)
        print(f"Plot saved to {plot_file}")
    
    # Return the sampled models and their scores
    return sampled_models, sampled_scores


def find_best_gaussian_fit(scores):
    """Find the best Gaussian distribution parameters that fit the data."""
    # This is a placeholder for a more sophisticated method
    # For now, we just use the mean and standard deviation of the data
    mean = np.mean(scores)
    std = np.std(scores)
    return mean, std


def plot_distributions(sampled_scores, target_mean, target_std, plot_file):
    """
    Plot the histogram of sampled model scores against the target Gaussian distribution.
    
    Args:
        sampled_scores: List of scores from the sampled models
        mean: Mean of the target Gaussian distribution
        std_dev: Standard deviation of the target Gaussian distribution
        plot_file: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot histogram of sampled scores
    plt.hist(sampled_scores, bins=20, alpha=0.6, density=True, 
             label=f'Sampled Models (n={len(sampled_scores)})')
    
    # Plot the target Gaussian distribution
    x = np.linspace(max(0, target_mean - 4*target_std), min(1, target_mean + 4*target_std), 1000)
    y = stats.norm.pdf(x, target_mean, target_std)
    plt.plot(x, y, 'r-', linewidth=2, label=f'Target Gaussian (μ={target_mean:.3f}, σ={target_std:.3f})')
    
    # Calculate the actual mean and std dev of the sampled scores
    actual_mean = np.mean(sampled_scores)
    actual_std = np.std(sampled_scores)
    plt.axvline(actual_mean, color='g', linestyle='--', 
                label=f'Actual Mean: {actual_mean:.3f}')
    plt.axvline(actual_mean + actual_std, color='g', linestyle=':', 
                label=f'Actual σ: {actual_std:.3f}')
    plt.axvline(actual_mean - actual_std, color='g', linestyle=':')
    
    # Calculate goodness of fit statistics
    _, p_value = stats.kstest(sampled_scores, 'norm', args=(actual_mean, actual_std))
    skewness = stats.skew(sampled_scores)
    kurtosis = stats.kurtosis(sampled_scores)
    
    # Add statistics to the plot
    plt.text(0.02, 0.95, f"Skewness: {skewness:.4f}\nKurtosis: {kurtosis:.4f}\np-value: {p_value:.4f}", 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title('Distribution of Sampled Model Scores')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Sample models to fit a Gaussian distribution of scores.')
    parser.add_argument('--response_matrix_file', type=str, 
                        default='filtered_8b_leaderboard_mmlu_pro_response_matrix_math.csv',
                        help='Path to the CSV file containing model responses')
    parser.add_argument('--avg_scores_file', type=str, 
                        default='8b_only_distribution_average_scores.csv',
                        help='Path to the CSV file containing average scores')
    parser.add_argument('--output_file', type=str, 
                        default='gaussian_sampled_models.csv',
                        help='Path to save the sampled models response matrix')
    parser.add_argument('--mean', type=float, default=None,
                        help='Mean of the target Gaussian distribution (default: auto-determined from data)')
    parser.add_argument('--std_dev', type=float, default=None,
                        help='Standard deviation of the target Gaussian distribution (default: auto-determined from data)')
    parser.add_argument('--num_models', type=int, default=None,
                        help='Number of models to sample (default: auto-determined from data)')
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable plotting')
    parser.add_argument('--model_col', type=str, default='Model',
                        help='Column name for model names in the average scores file')
    parser.add_argument('--score_col', type=str, default='Average Score',
                        help='Column name for scores in the average scores file')
    
    args = parser.parse_args()
    
    sample_models_for_gaussian(
        args.response_matrix_file,
        args.avg_scores_file,
        args.output_file,
        target_mean=args.mean,
        target_std=args.std_dev,
        num_models=args.num_models,
        plot=not args.no_plot,
        model_col=args.model_col,
        score_col=args.score_col
    )
