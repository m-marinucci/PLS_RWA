import pandas as pd
import numpy as np
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

# Read the results
results = pd.read_csv('simulation_results.csv')

# Overall ANOVA without method distinction
print("Overall ANOVA Results (without method distinction):")
model = ols('top_k_acc ~ C(n) + C(J) + C(magnitude) + C(noise) + C(rho) + C(data_type)', data=results).fit()
anova_table = anova_lm(model, typ=2)
print("\nANOVA Table:")
print(anova_table)

# Calculate effect sizes (partial eta-squared)
def partial_eta_squared(aov):
    aov['pes'] = aov['sum_sq'] / (aov['sum_sq'] + aov['sum_sq'].sum())
    return aov

anova_with_pes = partial_eta_squared(anova_table)
print("\nEffect Sizes (Partial Eta-Squared):")
print(anova_with_pes['pes'])

# Separate ANOVAs for each method
print("\nPLS-VIP ANOVA Results:")
pls_results = results[results['method'] == 'PLS-VIP']
pls_model = ols('top_k_acc ~ C(n) + C(J) + C(magnitude) + C(noise) + C(rho) + C(data_type)', data=pls_results).fit()
pls_anova = anova_lm(pls_model, typ=2)
print(partial_eta_squared(pls_anova))

print("\nRWA ANOVA Results:")
rwa_results = results[results['method'] == 'RWA']
rwa_model = ols('top_k_acc ~ C(n) + C(J) + C(magnitude) + C(noise) + C(rho) + C(data_type)', data=rwa_results).fit()
rwa_anova = anova_lm(rwa_model, typ=2)
print(partial_eta_squared(rwa_anova))

# Method comparison analysis
print("\nMethod Comparison Analysis:")
interaction_model = ols('top_k_acc ~ C(method) * (C(n) + C(J) + C(magnitude) + C(noise) + C(rho) + C(data_type))', data=results).fit()
interaction_anova = anova_lm(interaction_model, typ=2)
print(partial_eta_squared(interaction_anova))

# Save detailed results to file
with open('statistical_analysis_results.txt', 'w') as f:
    f.write("Statistical Analysis Results\n")
    f.write("==========================\n\n")
    f.write("Overall ANOVA Results (without method distinction):\n")
    f.write(str(anova_with_pes))
    f.write("\n\nPLS-VIP ANOVA Results:\n")
    f.write(str(partial_eta_squared(pls_anova)))
    f.write("\n\nRWA ANOVA Results:\n")
    f.write(str(partial_eta_squared(rwa_anova)))
    f.write("\n\nMethod Comparison Analysis:\n")
    f.write(str(partial_eta_squared(interaction_anova)))

# Create visualizations
plt.figure(figsize=(15, 10))
sns.boxplot(data=results, x='method', y='top_k_acc', hue='data_type')
plt.title('Performance Comparison by Method and Data Type')
plt.savefig('method_comparison.png')
plt.close()

# Create interaction plots
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

factors = ['n', 'J', 'magnitude', 'noise', 'rho', 'data_type']
for i, factor in enumerate(factors):
    sns.lineplot(data=results, x=factor, y='top_k_acc', hue='method', ax=axes[i])
    axes[i].set_title(f'Interaction with {factor}')

plt.tight_layout()
plt.savefig('interaction_plots.png')
plt.close() 