import pandas as pd
import numpy as np
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import os

# Create output directories if they don't exist
os.makedirs('docs/figures', exist_ok=True)
os.makedirs('docs/tables', exist_ok=True)

# Read the results
results = pd.read_csv('simulation_results.csv')

# 1. Basic Summary Statistics
basic_summary = results.groupby(['method', 'data_type'])['top_k_acc'].agg([
    'count', 'mean', 'std', 'min', 'max'
]).round(4)
basic_summary.to_csv('docs/tables/basic_summary.csv')
basic_summary.to_latex('docs/tables/basic_summary.tex')

# 2. Performance by Number of Predictors
j_summary = results.groupby(['method', 'J'])['top_k_acc'].agg([
    'count', 'mean', 'std'
]).round(4)
j_summary.to_csv('docs/tables/predictor_summary.csv')
j_summary.to_latex('docs/tables/predictor_summary.tex')

# 3. Performance by Correlation Level
rho_summary = results.groupby(['method', 'rho'])['top_k_acc'].agg([
    'count', 'mean', 'std'
]).round(4)
rho_summary.to_csv('docs/tables/correlation_summary.csv')
rho_summary.to_latex('docs/tables/correlation_summary.tex')

# 4. ANOVA Results
# Overall ANOVA
model = ols('top_k_acc ~ C(n) + C(J) + C(magnitude) + C(noise) + C(rho) + C(data_type)', data=results).fit()
anova_table = anova_lm(model, typ=2)
anova_table['partial_eta_sq'] = anova_table['sum_sq'] / (anova_table['sum_sq'] + anova_table['sum_sq'].sum())
anova_table = anova_table.round(4)
anova_table.to_csv('docs/tables/overall_anova.csv')
anova_table.to_latex('docs/tables/overall_anova.tex')

# Method-specific ANOVAs
for method in ['PLS-VIP', 'RWA']:
    method_data = results[results['method'] == method]
    method_model = ols('top_k_acc ~ C(n) + C(J) + C(magnitude) + C(noise) + C(rho) + C(data_type)', 
                      data=method_data).fit()
    method_anova = anova_lm(method_model, typ=2)
    method_anova['partial_eta_sq'] = method_anova['sum_sq'] / (method_anova['sum_sq'] + method_anova['sum_sq'].sum())
    method_anova = method_anova.round(4)
    method_anova.to_csv(f'docs/tables/{method.lower()}_anova.csv')
    method_anova.to_latex(f'docs/tables/{method.lower()}_anova.tex')

# 5. Performance distribution analysis
# Create performance categories based on percentile ranges
results['perf_category'] = 'Medium'  # default category
results.loc[results['top_k_acc'] <= results['top_k_acc'].quantile(0.25), 'perf_category'] = 'Low'
results.loc[results['top_k_acc'] >= results['top_k_acc'].quantile(0.75), 'perf_category'] = 'High'

# Cross-tabs for each factor
factors = ['J', 'rho', 'magnitude', 'noise', 'data_type']
for factor in factors:
    crosstab = pd.crosstab([results['method'], results[factor]], 
                          results['perf_category'], 
                          normalize='index').round(4)
    crosstab.to_csv(f'docs/tables/crosstab_{factor}.csv')
    crosstab.to_latex(f'docs/tables/crosstab_{factor}.tex')

# 6. Summary of interaction effects
interaction_model = ols('top_k_acc ~ C(method) * (C(n) + C(J) + C(magnitude) + C(noise) + C(rho) + C(data_type))', 
                       data=results).fit()
interaction_anova = anova_lm(interaction_model, typ=2)
interaction_anova['partial_eta_sq'] = interaction_anova['sum_sq'] / (interaction_anova['sum_sq'] + interaction_anova['sum_sq'].sum())
interaction_anova = interaction_anova.round(4)
interaction_anova.to_csv('docs/tables/interaction_effects.csv')
interaction_anova.to_latex('docs/tables/interaction_effects.tex')

# 7. Create a comprehensive summary markdown file
with open('docs/analysis_summary.md', 'w') as f:
    f.write('# Analysis Summary\n\n')
    
    f.write('## Basic Performance Statistics\n')
    f.write(basic_summary.to_markdown())
    f.write('\n\n')
    
    f.write('## Effect of Number of Predictors\n')
    f.write(j_summary.to_markdown())
    f.write('\n\n')
    
    f.write('## Effect of Correlation Structure\n')
    f.write(rho_summary.to_markdown())
    f.write('\n\n')
    
    f.write('## ANOVA Results\n')
    f.write(anova_table.to_markdown())
    f.write('\n\n')
    
    f.write('## Interaction Effects\n')
    f.write(interaction_anova.to_markdown())
    f.write('\n\n')
    
    # Add performance distribution information
    f.write('## Performance Distribution\n')
    f.write('Performance categories are defined as:\n')
    f.write(f'- Low: Bottom 25% (≤ {results["top_k_acc"].quantile(0.25):.4f})\n')
    f.write(f'- Medium: Middle 50%\n')
    f.write(f'- High: Top 25% (≥ {results["top_k_acc"].quantile(0.75):.4f})\n\n')

# Move visualization outputs to docs/figures
import shutil
for file in ['finding1_predictors.png', 'finding2_correlation.png', 'finding3_datatype.png',
             'finding4_overall.png', 'finding5_interactions.png', 'finding6_heatmaps.png',
             'summary_plot.png']:
    if os.path.exists(file):
        shutil.move(file, f'docs/figures/{file}')

print("Summary tables and figures have been generated and organized in the docs directory.") 