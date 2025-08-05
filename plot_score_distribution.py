#!/usr/bin/env python3
"""
Script to plot the distribution of average scores and compare with a Gaussian distribution
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

def extract_scores(leaderboard_data):
    """Extract average scores from leaderboard data."""
    scores = []
    
    for entry in tqdm(leaderboard_data, desc="Extracting scores"):
        if 'model' in entry and 'average_score' in entry['model']:
            score = entry['model']['average_score']
            if score is not None and isinstance(score, (int, float)):
                scores.append(score)
    
    return np.array(scores)

def plot_score_distribution(scores, output_file=None):
    """Plot histogram of scores and compare with Gaussian."""
    plt.figure(figsize=(12, 8))
    
    # Plot histogram with density=True to get probability density
    n, bins, patches = plt.hist(scores, bins=50, density=True, alpha=0.7, 
                               label='Average Score Distribution')
    
    # Fit a normal distribution to the data
    mu, sigma = stats.norm.fit(scores)
    
    # Plot the PDF (probability density function)
    x = np.linspace(min(scores), max(scores), 100)
    pdf = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, pdf, 'r-', linewidth=2, label=f'Fitted Gaussian: μ={mu:.2f}, σ={sigma:.2f}')
    
    # Calculate goodness of fit with Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.kstest(scores, 'norm', args=(mu, sigma))
    
    # Add title and labels
    plt.title(f'Distribution of LLM Average Scores\nKS test: statistic={ks_statistic:.4f}, p-value={p_value:.4f}', 
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
    parser = argparse.ArgumentParser(description="Plot distribution of average scores and compare with Gaussian.")
    parser.add_argument("--leaderboard", default="open_llm_leaderboard_data.json", help="Leaderboard data file")
    parser.add_argument("--output", default="score_distribution.png", help="Output image file")
    args = parser.parse_args()
    
    # Load leaderboard data
    print(f"Loading leaderboard data from {args.leaderboard}...")
    leaderboard_data = load_json_file(args.leaderboard)
    print(f"Found {len(leaderboard_data)} entries in leaderboard.")
    
    # Extract scores
    scores = extract_scores(leaderboard_data)
    print(f"Extracted {len(scores)} valid scores.")
    
    # Plot the distribution
    print("Generating plot...")
    plot_score_distribution(scores, args.output)
    
    # Additional analysis: skewness and kurtosis
    skewness = stats.skew(scores)
    kurtosis = stats.kurtosis(scores)
    print(f"Distribution Skewness: {skewness:.4f}")
    print(f"Distribution Kurtosis: {kurtosis:.4f}")
    print("(For a perfect Gaussian: Skewness=0, Kurtosis=0)")

if __name__ == "__main__":
    main()
