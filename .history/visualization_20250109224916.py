import pandas as pd
import matplotlib.pyplot as plt

def plot_comparison(results_df, save_path='comparison_plot.png'):
    """Create a comparison plot between RWA and PLS performance."""
    # Set style
    plt.style.use('seaborn')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for plotting
    methods = ['RWA', 'PLS']
    metrics = ['Top-1', 'Top-2', 'Top-3']
    
    data = {
        'RWA': [results_df['top1_rwa'].mean(), results_df['top2_rwa'].mean(), results_df['top3_rwa'].mean()],
        'PLS': [results_df['top1_pls'].mean(), results_df['top2_pls'].mean(), results_df['top3_pls'].mean()]
    }
    
    # Set width of bars and positions
    barWidth = 0.35
    r1 = range(len(metrics))
    r2 = [x + barWidth for x in r1]
    
    # Create bars
    plt.bar(r1, data['RWA'], width=barWidth, label='RWA', color='skyblue', alpha=0.8)
    plt.bar(r2, data['PLS'], width=barWidth, label='PLS', color='lightgreen', alpha=0.8)
    
    # Add error bars
    plt.errorbar(r1, data['RWA'], 
                yerr=[results_df['top1_rwa'].std(), results_df['top2_rwa'].std(), results_df['top3_rwa'].std()],
                fmt='none', color='black', capsize=5)
    plt.errorbar(r2, data['PLS'], 
                yerr=[results_df['top1_pls'].std(), results_df['top2_pls'].std(), results_df['top3_pls'].std()],
                fmt='none', color='black', capsize=5)
    
    # Add labels and title
    plt.xlabel('Accuracy Metric')
    plt.ylabel('Mean Accuracy')
    plt.title('Performance Comparison: RWA vs PLS')
    plt.xticks([r + barWidth/2 for r in range(len(metrics))], metrics)
    plt.legend()
    
    # Set y-axis to start from 0
    plt.ylim(0, 1.1)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
