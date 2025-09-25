import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
import argparse
def main():
    parser = argparse.ArgumentParser(description='Compare average scores and Whole-bank Ability')
    parser.add_argument('benchmark_name', type=str, help='Benchmark name')
    args = parser.parse_args()

    if args.benchmark_name == 'hellaswag' or args.benchmark_name == 'winogrande' or args.benchmark_name == 'truthfulqa':
        avg_scores_file = f'select_model/gaussian_sampled_{args.benchmark_name}_response_matrix_train_with_scores.csv'
    else:
        avg_scores_file = f'select_model/gaussian_sampled_{args.benchmark_name}_response_matrix.csv'
        

    
    theta_whole_file = f'{args.benchmark_name}/irt_person_scores_WLE_SE.csv'
    

    # Read the two CSV files
    avg_scores_df = pd.read_csv(avg_scores_file)
    theta_whole_df = pd.read_csv(theta_whole_file)
    
    # For arc and gsm8k, we need to calculate the average scores from the response matrix
    if args.benchmark_name == 'arc' or args.benchmark_name == 'gsm8k':
        # Check if 'Average Score' is already in the columns
        if 'Average Score' not in avg_scores_df.columns:
            # Assuming the response matrix has model names in the first column and the rest are responses
            # First column is model name, remaining columns are item responses (1 for correct, 0 for incorrect)
            model_name_col = avg_scores_df.columns[0]
            response_columns = avg_scores_df.columns[1:]
            
            # Calculate the average score for each model (row mean excluding the model name column)
            avg_scores_df['Average Score'] = avg_scores_df[response_columns].mean(axis=1)
            
            # Ensure the first column is named 'Model_Name' for consistency
            avg_scores_df = avg_scores_df.rename(columns={model_name_col: 'Model_Name'})
    # avg_scores_df = pd.read_csv('gaussian_distribution_scores.csv')
    # theta_whole_df = pd.read_csv('irt_person_scores_gaussian_sampled_300_WLE_SE.csv')
    # avg_scores_df = pd.read_csv('8b_only_distribution_average_scores.csv')
    # theta_whole_df = pd.read_csv('irt_person_scores_112_8b_only_WLE.csv')
    # Standardize column names
    # avg_scores_df = avg_scores_df.rename(columns={'Model': 'Model_Name'})
    avg_scores_df = avg_scores_df.rename(columns={'Unnamed: 0': 'Model_Name'})
    avg_scores_df = avg_scores_df.rename(columns={'avg_score': 'Average Score'})
    # Merge the dataframes on model name
    # First, create copy of dataframes with standardized model names for merging
    avg_scores_copy = avg_scores_df.copy()
    theta_whole_copy = theta_whole_df.copy()
    
    # Get the common models between the two datasets
    common_models = set(avg_scores_copy['Model_Name']).intersection(set(theta_whole_copy['Model_Name']))
    
    if len(common_models) == 0:
        print("No common models found with exact name match. Trying to match by final part of model name...")
        
        # Extract the final part of the model names (after the last slash)
        avg_scores_copy['Model_Short'] = avg_scores_copy['Model_Name'].apply(lambda x: x.split('/')[-1])
        theta_whole_copy['Model_Short'] = theta_whole_copy['Model_Name'].apply(lambda x: x.split('/')[-1])
        
        # Now try to merge on these shortened names
        merged_df = pd.merge(
            avg_scores_copy[['Model_Name', 'Model_Short', 'Average Score']], 
            theta_whole_copy[['Model_Name', 'Model_Short', 'Theta_WLE']], 
            on='Model_Name', 
            how='inner',
            suffixes=('_avg', '_theta')
        )
        
        if len(merged_df) == 0:
            print("Still no matches found. The model naming conventions are too different.")
            print("Here are some examples from each file:")
            print("\nAverage Scores file:")
            print(avg_scores_df['Model_Name'].head())
            print("\nTheta WLE file:")
            print(theta_whole_df['Model_Name'].head())
            return
    else:
        # Filter both dataframes to only include common models
        avg_scores_filtered = avg_scores_copy[avg_scores_copy['Model_Name'].isin(common_models)]
        theta_whole_filtered = theta_whole_copy[theta_whole_copy['Model_Name'].isin(common_models)]
        
        # Merge the filtered dataframes
        merged_df = pd.merge(
            avg_scores_filtered[['Model_Name', 'Average Score']], 
            theta_whole_filtered[['Model_Name', 'Theta_WLE']], 
            on='Model_Name'
        )
    
    # Calculate rankings for both metrics - starting from 1
    merged_df['Avg_Score_Rank'] = merged_df['Average Score'].rank(ascending=False)
    # Rename Theta_WLE to Theta_Whole for consistency
    merged_df = merged_df.rename(columns={'Theta_WLE': 'Theta_Whole'})
    merged_df['Theta_Whole_Rank'] = merged_df['Theta_Whole'].rank(ascending=False)
    
    # Make sure ranks start at 1, not 0
    merged_df['Avg_Score_Rank'] = merged_df['Avg_Score_Rank'].astype(int)
    merged_df['Theta_Whole_Rank'] = merged_df['Theta_Whole_Rank'].astype(int)
    
    # Calculate rank differences
    merged_df['Rank_Difference'] = merged_df['Avg_Score_Rank'] - merged_df['Theta_Whole_Rank']
    
    # Sort by Average Score rank
    merged_df_sorted_avg = merged_df.sort_values('Avg_Score_Rank')
    
    # Print the top models by Average Score
    print("\nTop 10 Models by Average Score:")
    print(merged_df_sorted_avg[['Model_Name', 'Average Score', 'Avg_Score_Rank', 'Theta_Whole', 'Theta_Whole_Rank', 'Rank_Difference']].head(10).to_string(index=False))
    
    # Sort by Theta WLE rank
    merged_df_sorted_theta = merged_df.sort_values('Theta_Whole_Rank')
    
    # Print the top models by Theta WLE
    print("\nTop 10 Models by Theta Whole:")
    print(merged_df_sorted_theta[['Model_Name', 'Theta_Whole', 'Theta_Whole_Rank', 'Average Score', 'Avg_Score_Rank', 'Rank_Difference']].head(10).to_string(index=False))
    
    # Calculate correlations
    spearman_corr, spearman_p = spearmanr(merged_df['Average Score'], merged_df['Theta_Whole'])
    kendall_corr, kendall_p = kendalltau(merged_df['Average Score'], merged_df['Theta_Whole'])
    
    print(f"\nCorrelation between Average Score and Theta Whole:")
    print(f"Spearman correlation: {spearman_corr:.2f} (p-value: {spearman_p:.2f})")
    print(f"Kendall's Tau correlation: {kendall_corr:.2f} (p-value: {kendall_p:.2f})")
    
    # Calculate correlation between ranks
    spearman_rank_corr, spearman_rank_p = spearmanr(merged_df['Avg_Score_Rank'], merged_df['Theta_Whole_Rank'])
    kendall_rank_corr, kendall_rank_p = kendalltau(merged_df['Avg_Score_Rank'], merged_df['Theta_Whole_Rank'])
    
    print(f"\nCorrelation between Average Score Rank and Theta Whole Rank:")
    print(f"Spearman correlation: {spearman_rank_corr:.2f} (p-value: {spearman_rank_p:.2f})")
    print(f"Kendall's Tau correlation: {kendall_rank_corr:.2f} (p-value: {kendall_rank_p:.2f})")
    
    # Count models with large rank differences
    large_diff_count = len(merged_df[abs(merged_df['Rank_Difference']) > 10])
    print(f"\nNumber of models with absolute rank difference > 10: {large_diff_count} (out of {len(merged_df)})")
    
    # Define dataset-specific title based on benchmark name
    dataset_titles = {
        'gsm8k': 'GSM8K',
        'hellaswag': 'HellaSwag',
        'winogrande': 'WinoGrande',
        'truthfulqa': 'TruthfulQA',
        'arc': 'ARC'
    }
    
    dataset_title = dataset_titles.get(args.benchmark_name, args.benchmark_name.upper())
    
    # Create a combined figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(28, 14))
    
    # Define outliers (models with large rank differences)
    good_models = merged_df[abs(merged_df['Rank_Difference']) <= 2]
    outliers = merged_df[abs(merged_df['Rank_Difference']) > 1500]
    non_outliers = merged_df[abs(merged_df['Rank_Difference']) <= 500]
    
    # Save outlier and good model data
    outliers.to_csv(f'{args.benchmark_name}/outliers.csv', index=False)
    good_models.to_csv(f'{args.benchmark_name}/good_models.csv', index=False)
    
    # Create a figure with extra space at the top for titles
    plt.figure(figsize=(30, 16))  # Slightly taller figure
    plt.clf()  # Clear the current figure
    
    # Create a new figure and adjust spacing at the top
    fig, axes = plt.subplots(1, 2, figsize=(30, 16))
    plt.subplots_adjust(top=0.70)  # Much more space at top
    
    # Add overall title with large font and ample top margin
    # fig.suptitle(dataset_title, fontsize=80, fontweight='bold', y=0.85)
    
    # Add correlation text below the title with clear spacing
    # fig.text(0.5, 0.78, f'Spearman Corr: {spearman_corr:.4f}   Kendall Corr: {kendall_corr:.2f}', 
    #          ha='center', va='center', fontsize=65, fontweight='bold')
    
    # First subplot: Score Comparison
    axes[0].scatter(non_outliers['Average Score'], non_outliers['Theta_Whole'], color='#e8a81f', alpha=0.3, s=20)
    # No outlier highlighting
    
    # More mathematical subtitle using LaTeX
    # axes[0].set_title(r'Whole-bank Ability vs. Average Score', fontsize=40)
    # axes[0].set_xlabel('Average Score', fontsize=40)
    # axes[0].set_ylabel(r'Whole-bank Ability', fontsize=40)
    axes[0].tick_params(axis='both', which='major', labelsize=40)
    axes[0].grid(True, linestyle='--', alpha=1)
    print("⚠️", args.benchmark_name)
    # Set specific axis limits for arc benchmark
    if args.benchmark_name == 'arc':
        axes[0].set_xlim(0.35, 0.75)
        axes[0].set_ylim(-2.5, 3)
    elif args.benchmark_name == 'hellaswag':
        axes[0].set_xlim(0.6, 0.9)
        axes[0].set_ylim(-2.5, 2.5)
    elif args.benchmark_name == 'winogrande':
        axes[0].set_xlim(0.6, 0.9)
        axes[0].set_ylim(-2, 5)
    # Second subplot: Rank Comparison
    axes[1].scatter(non_outliers['Avg_Score_Rank'], non_outliers['Theta_Whole_Rank'], color='#55b3e8', alpha=0.3, s=20)
    
    # No outlier highlighting in rank comparison
    
    # Add a more prominent diagonal line representing perfect rank correlation
    max_rank = max(merged_df['Avg_Score_Rank'].max(), merged_df['Theta_Whole_Rank'].max())
    axes[1].plot([1, max_rank], [1, max_rank], color='red', linestyle='--', linewidth=1.5)
    
    # More mathematical subtitle using LaTeX
    # axes[1].set_title(r'$\mathrm{Rank}(\text{Whole-bank Ability})$ vs. $\mathrm{Rank}(\text{Average Score})$', fontsize=40)
    # axes[1].set_xlabel('Average Score Rank (1 = best)', fontsize=40)
    # axes[1].set_ylabel(r'Whole-bank Ability Rank (1 = best)', fontsize=40)
    axes[1].tick_params(axis='both', which='major', labelsize=40)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Standardize tick density for better visual balance
    # Get current y-axis limits
    y0_min, y0_max = axes[0].get_ylim()
    
    # Adjust number of ticks for more balanced appearance
    from matplotlib.ticker import MaxNLocator
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=8))
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=8))
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=6))
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=6))
    
    # Force the right plot's axes to start exactly at 1
    axes[1].set_xlim(1, axes[1].get_xlim()[1])
    axes[1].set_ylim(1, axes[1].get_ylim()[1])
    
    # Force matplotlib to include specific ticks with 1 as the first tick
    from matplotlib.ticker import FixedLocator
    # Calculate reasonable tick positions with 1 as first tick
    max_x = axes[1].get_xlim()[1]
    max_y = axes[1].get_ylim()[1]
    
    # Create tick marks that always include 1
    x_ticks = [1]
    y_ticks = [1]
    
    # Calculate appropriate tick spacing to ensure 1 is visible
    x_step = max(int(max_x / 5), 1)
    for i in range(1, 6):
        tick = 1 + i * x_step  # Start from 1 plus steps
        if tick <= max_x:
            x_ticks.append(tick)
    
    y_step = max(int(max_y / 5), 1)
    for i in range(1, 6):
        tick = 1 + i * y_step  # Start from 1 plus steps
        if tick <= max_y:
            y_ticks.append(tick)
    
    # Set the ticks with 1 as first tick
    axes[1].xaxis.set_major_locator(FixedLocator(x_ticks))
    axes[1].yaxis.set_major_locator(FixedLocator(y_ticks))
    
    # Adjust layout and save the combined figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    plt.savefig(f'{args.benchmark_name}/combined_comparison.png', dpi=300)
    print(f"\nCombined plot saved as '{args.benchmark_name}/combined_comparison.png'")
    
    # Also save individual plots for backward compatibility
    # Score comparison plot
    plt.figure(figsize=(10, 8))
    plt.scatter(non_outliers['Average Score'], non_outliers['Theta_Whole'], color='#e8a81f', alpha=0.3, s=20)
    
    # No outlier highlighting in individual plot
    # plt.title(r'Whole-bank Ability vs. Average Score', fontsize=14)
    # plt.xlabel('Average Score', fontsize=14)
    # plt.ylabel(r'Whole-bank Ability', fontsize=14)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.grid(True, linestyle='--', alpha=1)
    # plt.text(0.05, 0.95, f'Spearman Corr: {spearman_corr:.2f}', transform=plt.gca().transAxes, fontsize=14)
    # plt.text(0.05, 0.90, f'Kendall Corr: {kendall_corr:.2f}', transform=plt.gca().transAxes, fontsize=14)
    
    # Set specific axis limits for arc benchmark in individual plot
    if args.benchmark_name == 'arc':
        plt.xlim(0.35, 0.75)
        plt.ylim(-2.5, 3)
    elif args.benchmark_name == 'HellaSwag':
        plt.xlim(0.6, 1.0)
        plt.ylim(-3.5, 2.5)
    
    plt.tight_layout()
    plt.savefig(f'{args.benchmark_name}/scores_comparison.png')
    
    # Rank comparison plot
    plt.figure(figsize=(10, 8))
    plt.scatter(non_outliers['Avg_Score_Rank'], non_outliers['Theta_Whole_Rank'], color='#55b3e8', alpha=0.3, s=20)
    
    # No outlier highlighting in individual rank plot
                   
    max_rank = max(merged_df['Avg_Score_Rank'].max(), merged_df['Theta_Whole_Rank'].max())
    plt.plot([1, max_rank], [1, max_rank], color='red', linestyle='--', linewidth=1.5)
    # Force both axes to start exactly at 1
    plt.xlim(1, plt.gca().get_xlim()[1])
    plt.ylim(1, plt.gca().get_ylim()[1])
    
    # Force matplotlib to include specific ticks with 1 as the first tick
    from matplotlib.ticker import FixedLocator
    # Calculate reasonable tick positions with 1 as first tick
    max_x = plt.gca().get_xlim()[1]
    max_y = plt.gca().get_ylim()[1]
    
    # Create tick marks that always include 1
    x_ticks = [1]
    y_ticks = [1]
    
    # Calculate appropriate tick spacing to ensure 1 is visible
    x_step = max(int(max_x / 5), 1)
    for i in range(1, 6):
        tick = 1 + i * x_step  # Start from 1 plus steps
        if tick <= max_x:
            x_ticks.append(tick)
    
    y_step = max(int(max_y / 5), 1)
    for i in range(1, 6):
        tick = 1 + i * y_step  # Start from 1 plus steps
        if tick <= max_y:
            y_ticks.append(tick)
    
    # Set the ticks with 1 as first tick
    plt.gca().xaxis.set_major_locator(FixedLocator(x_ticks))
    plt.gca().yaxis.set_major_locator(FixedLocator(y_ticks))
    # plt.title(r'$\mathrm{Rank}(\text{Whole-bank Ability})$ vs. $\mathrm{Rank}(\text{Average Score})$', fontsize=40)
    # plt.xlabel('Average Score Rank (1 = best)', fontsize=40)
    # plt.ylabel(r'$\mathrm{Rank}(\text{Whole-bank Ability})$', fontsize=40)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.text(0.05, 0.95, f'Spearman Rank Corr: {spearman_rank_corr:.2f}', transform=plt.gca().transAxes, fontsize=40)
    # plt.text(0.05, 0.90, f'Kendall Rank Corr: {kendall_rank_corr:.2f}', transform=plt.gca().transAxes, fontsize=40)
    from matplotlib.ticker import MaxNLocator
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.tight_layout()
    plt.savefig(f'{args.benchmark_name}/rank_comparison.png')
    print(f"Individual plots also saved as '{args.benchmark_name}/scores_comparison.png' and '{args.benchmark_name}/rank_comparison.png'")
    print(f"\nOutliers saved as '{args.benchmark_name}/outliers.csv'")
    print(f"\nGood models saved as '{args.benchmark_name}/good_models.csv'")

if __name__ == "__main__":
    main()
