import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read results
results = pd.read_csv('simulation_results.csv')

# Set style parameters
plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 10
colors = ['#2ecc71', '#e74c3c']  # Green for continuous, Red for discrete

# Create figure with subplots
fig, axes = plt.subplots(2, 3)
fig.suptitle('Interaction Effects with Data Type', fontsize=14, y=1.02)

# Plot settings
methods = ['PLS-VIP', 'RWA']
data_types = ['continuous', 'discrete']
factors = ['n', 'J', 'rho']
factor_names = ['Sample Size', 'Number of Predictors', 'Correlation']

# Create plots for each method
for method_idx, method in enumerate(methods):
    method_data = results[results['method'] == method]
    
    # Create interaction plots for each factor
    for factor_idx, (factor, factor_name) in enumerate(zip(factors, factor_names)):
        ax = axes[method_idx, factor_idx]
        
        # Calculate means for each combination
        means = method_data.groupby([factor, 'data_type'])['top_k_acc'].mean().unstack()
        
        # Plot lines
        for dtype_idx, dtype in enumerate(data_types):
            ax.plot(means.index, means[dtype], marker='o', 
                   label=dtype.capitalize(), color=colors[dtype_idx],
                   linewidth=2, markersize=8)
        
        # Customize plot
        ax.set_xlabel(factor_name)
        ax.set_ylabel('Top-1 Accuracy' if factor_idx == 0 else '')
        if factor_idx == 0:
            ax.set_title(f'{method}')
        if factor_idx == 2:  # Legend only for last column
            ax.legend(title='Data Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set appropriate x-ticks
        if factor == 'rho':
            ax.set_xticks([0.0, 0.5, 0.95])
        
        # Set y-axis limits consistently
        ax.set_ylim(0, max(means.max().max() * 1.1, 0.4))
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig('interaction_plots.png', dpi=300, bbox_inches='tight')
plt.close()

# Create overall bar plot
plt.figure(figsize=(10, 6))
overall_means = results.groupby(['method', 'data_type'])['top_k_acc'].mean().unstack()

# Plot bars
bar_width = 0.35
x = np.arange(len(methods))
plt.bar(x - bar_width/2, overall_means['continuous'].values, bar_width, 
        label='Continuous', color=colors[0])
plt.bar(x + bar_width/2, overall_means['discrete'].values, bar_width,
        label='Discrete', color=colors[1])

# Customize plot
plt.xlabel('Method')
plt.ylabel('Top-1 Accuracy')
plt.title('Overall Performance by Method and Data Type')
plt.xticks(x, methods)
plt.legend(title='Data Type')
plt.grid(True, linestyle='--', alpha=0.7)

# Add value labels on bars
for i, method in enumerate(methods):
    plt.text(i - bar_width/2, overall_means.loc[method, 'continuous'], 
             f'{overall_means.loc[method, "continuous"]:.3f}', 
             ha='center', va='bottom')
    plt.text(i + bar_width/2, overall_means.loc[method, 'discrete'],
             f'{overall_means.loc[method, "discrete"]:.3f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig('overall_performance.png', dpi=300, bbox_inches='tight')
plt.close() 