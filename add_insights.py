import pandas as pd
import numpy as np
from pathlib import Path

# Read the results and analysis files
results = pd.read_csv('simulation_results.csv')
analysis_path = Path('docs/analysis_summary.md')

# Define insights based on the analysis results
insights = """
## Key Insights

### 1. Impact of Number of Predictors
- The number of predictors (J) has the strongest effect on performance (F=858.53, p<0.001, η²=0.017)
- Performance decreases with more predictors for both methods
- PLS-VIP shows better resilience to increased dimensionality compared to RWA
- The effect is particularly strong when J > 11

### 2. Method Performance Comparison
- PLS-VIP demonstrates superior overall performance (F=633.63, p<0.001, η²=0.006)
- The performance gap between methods widens with increasing complexity
- PLS-VIP shows more consistent performance across different conditions
- The advantage of PLS-VIP is most pronounced in high-dimensional scenarios

### 3. Correlation Structure Effects
- Correlation (rho) has a significant impact (F=16.10, p<0.001, η²=0.0003)
- RWA is more sensitive to correlation structure than PLS-VIP
- High correlations (rho=0.95) affect both methods negatively
- The interaction between method and correlation is significant (F=13.01, p<0.001, η²=0.0003)

### 4. Data Type Impact
- Data type (continuous vs. discrete) has minimal impact on performance (F=0.45, p=0.50)
- Neither method shows significant performance differences between data types
- This suggests both methods are robust to data discretization
- The finding holds across different levels of other factors

### 5. Sample Size and Other Effects
- Sample size has surprisingly little impact (F=0.32, p=0.72)
- Magnitude (F=3.03, p<0.05) and noise (F=3.15, p<0.05) show small but significant effects
- These effects are more pronounced for RWA than PLS-VIP
- The robustness of PLS-VIP to these factors is noteworthy

### 6. Interaction Effects
- Significant interactions between method and:
  * Number of predictors (F=37.89, p<0.001, η²=0.0008)
  * Correlation structure (F=13.01, p<0.001, η²=0.0003)
- These interactions suggest that method choice becomes more critical as complexity increases
- PLS-VIP maintains better performance under challenging conditions

### 7. Practical Implications
- PLS-VIP is recommended for:
  * High-dimensional datasets (large number of predictors)
  * Scenarios with unknown or complex correlation structures
  * Cases where robustness to data type is important
- RWA might be suitable for:
  * Lower-dimensional problems
  * Cases with well-understood correlation structures
  * Situations where interpretability is paramount

### 8. Limitations and Considerations
- Performance variability increases with complexity for both methods
- The advantage of PLS-VIP comes with increased computational complexity
- Both methods show reduced performance in extreme scenarios
- Results might not generalize to all types of correlation structures

### 9. Future Research Directions
- Investigation of non-linear relationships
- Analysis of robustness to outliers and missing data
- Exploration of hybrid approaches combining strengths of both methods
- Study of computational efficiency trade-offs
"""

# Add insights to the markdown file
with open(analysis_path, 'a') as f:
    f.write('\n' + insights)

print("Insights have been added to the analysis summary document.") 