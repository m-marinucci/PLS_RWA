import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the results
results = pd.read_csv('simulation_results.csv')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. Impact of number of predictors (J)
plt.figure(figsize=(10, 6))
sns.boxplot(data=results, x='J', y='top_k_acc', hue='method')
plt.title('Impact of Number of Predictors on Performance')
plt.xlabel('Number of Predictors (J)')
plt.ylabel('Top-k Accuracy')
plt.savefig('finding1_predictors.png')
plt.close()

# 2. Sensitivity to correlation structure
plt.figure(figsize=(10, 6))
sns.lineplot(data=results, x='rho', y='top_k_acc', hue='method', err_style='band')
plt.title('Method Sensitivity to Correlation Structure')
plt.xlabel('Correlation (ρ)')
plt.ylabel('Top-k Accuracy')
plt.savefig('finding2_correlation.png')
plt.close()

# 3. Data type effect
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.violinplot(data=results[results['method']=='PLS-VIP'], x='data_type', y='top_k_acc', ax=ax1)
ax1.set_title('PLS-VIP: Performance by Data Type')
sns.violinplot(data=results[results['method']=='RWA'], x='data_type', y='top_k_acc', ax=ax2)
ax2.set_title('RWA: Performance by Data Type')
plt.tight_layout()
plt.savefig('finding3_datatype.png')
plt.close()

# 4. Overall method performance
plt.figure(figsize=(10, 6))
sns.violinplot(data=results, x='method', y='top_k_acc')
plt.title('Overall Performance Comparison Between Methods')
plt.ylabel('Top-k Accuracy')
plt.savefig('finding4_overall.png')
plt.close()

# 5. Method interactions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Interaction with number of predictors
sns.pointplot(data=results, x='J', y='top_k_acc', hue='method', ax=ax1)
ax1.set_title('Method × Number of Predictors Interaction')
ax1.set_xlabel('Number of Predictors (J)')

# Interaction with correlation
sns.pointplot(data=results, x='rho', y='top_k_acc', hue='method', ax=ax2)
ax2.set_title('Method × Correlation Interaction')
ax2.set_xlabel('Correlation (ρ)')

plt.tight_layout()
plt.savefig('finding5_interactions.png')
plt.close()

# Additional: Factor effects heatmap
# Calculate mean performance for each combination of factors
pivot_j = pd.pivot_table(results, values='top_k_acc', 
                        index='method', columns='J', 
                        aggfunc='mean')
pivot_rho = pd.pivot_table(results, values='top_k_acc', 
                          index='method', columns='rho', 
                          aggfunc='mean')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Heatmap for J
sns.heatmap(pivot_j, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax1)
ax1.set_title('Mean Performance by Number of Predictors')

# Heatmap for correlation
sns.heatmap(pivot_rho, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax2)
ax2.set_title('Mean Performance by Correlation Level')

plt.tight_layout()
plt.savefig('finding6_heatmaps.png')
plt.close()

# Create a summary plot combining key insights
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 3)

# Main effects
ax1 = fig.add_subplot(gs[0, 0])
sns.boxplot(data=results, x='method', y='top_k_acc', ax=ax1)
ax1.set_title('Overall Performance')

# J effect
ax2 = fig.add_subplot(gs[0, 1:])
sns.lineplot(data=results, x='J', y='top_k_acc', hue='method', 
            style='data_type', markers=True, dashes=False, ax=ax2)
ax2.set_title('Impact of Predictors and Data Type')

# Correlation effect
ax3 = fig.add_subplot(gs[1, :2])
sns.lineplot(data=results, x='rho', y='top_k_acc', hue='method', 
            size='J', sizes=(1, 4), ax=ax3)
ax3.set_title('Correlation and Predictor Interaction')

# Performance distribution
ax4 = fig.add_subplot(gs[1, 2])
sns.kdeplot(data=results, x='top_k_acc', hue='method', ax=ax4)
ax4.set_title('Performance Distribution')

plt.tight_layout()
plt.savefig('summary_plot.png')
plt.close() 