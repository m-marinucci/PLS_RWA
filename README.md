# PLS-VIP vs RWA Comparison Study

A statistical simulation project comparing **PLS-VIP** (Partial Least Squares - Variable Importance in Projection) and **RWA** (Relative Weight Analysis) methods for identifying important predictors in regression models.

## 📄 Research Paper

**[Comparative Analysis of PLS-VIP and RWA Methods: A Comprehensive Simulation Study](paper/main.pdf)**

This repository contains the complete implementation and reproducible code for our research paper that presents a comprehensive comparison between PLS-VIP and RWA methods through extensive simulation studies. The paper investigates method performance across various conditions including different sample sizes, predictor counts, correlation structures, and data types, providing practical guidelines for method selection in applied research.

**Key findings from the paper:**
- Significant performance differences between methods in high-dimensional scenarios
- Impact of correlation structures on method effectiveness
- Practical recommendations for method selection based on data characteristics
- Comprehensive statistical analysis with ANOVA and cross-tabulation results

## Table of Contents

- [📄 Research Paper](#-research-paper)
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Parameter Space](#parameter-space)
- [Architecture](#architecture)
- [Output Files](#output-files)
- [Methods](#methods)
- [Reproducing Paper Results](#reproducing-paper-results)

## Overview

This simulation study evaluates two competing methods for variable importance assessment:

- **PLS-VIP**: Uses cross-decomposition to score variable importance
- **RWA**: Johnson's Relative Weight Analysis using transformation matrices

The simulation tests performance across multiple parameter combinations using Monte Carlo experiments to determine which method performs better under different conditions.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running the Simulation

```bash
python main.py
```

This will:
- Run 100 replications across multiple parameter combinations
- Save results to `simulation_results.csv`
- Display performance analysis
- Generate data that supports the findings in our [research paper](paper/main.pdf)

## Parameter Space

The simulation tests across:
- **Sample sizes**: 100-500
- **Number of predictors**: 10-50
- **Correlation levels**: 0.1-0.5
- **Effect magnitudes**: 0.1-0.5
- **Noise ratios**: 0.1-0.5
- **Importance proportion α**: 0.3 (default)

## Architecture

- **`factors.py`**: Statistical core implementing both methods and data generation
- **`main.py`**: Simulation engine orchestrating Monte Carlo experiments
- **`visualization.py`**: Performance visualization utilities
- **`logger.py`**: Logging configuration

## Output Files

- **`simulation_results.csv`**: Raw simulation data
- **`detailed_results.csv`**: Aggregated performance metrics
- **`fractional_factorial_results.csv`**: Fractional factorial design results
- **`comparison_plot.png`** & **`performance_comparison.png`**: Visual comparisons
- **`docs/`**: Analysis summaries and additional figures
- **`paper/`**: Complete research paper with LaTeX source, figures, and tables

## Configuration

Key parameters in `main.py`:
- `alpha`: Proportion of variables that are important (default 0.3)
- `randomize_important`: Whether to randomize important variables across replications
- Results are saved incrementally to prevent data loss

## Methods

### PLS-VIP
Uses partial least squares regression to decompose the predictor space and calculates Variable Importance in Projection scores based on the contribution of each variable to the latent components.

### RWA (Relative Weight Analysis)
Johnson's method that uses transformation matrices to partition the variance in the outcome variable among correlated predictors, providing relative importance weights.

## Reproducing Paper Results

To reproduce the exact results presented in the research paper:

1. **Run the full simulation study:**
   ```bash
   python main.py
   ```

2. **Generate statistical analysis:**
   ```bash
   python statistical_analysis.py
   ```

3. **Create visualizations:**
   ```bash
   python visualize_results.py
   ```

4. **Reproduce all analyses at once:**
   ```bash
   bash reproduce_results.sh
   ```

The paper's figures and tables are automatically generated from the simulation results and saved in the `docs/` directory. All statistical analyses, including ANOVA results and cross-tabulations presented in the paper, can be reproduced using the provided scripts.

**Note:** The full simulation may take several hours to complete. For quick testing, modify the `n_reps` parameter in `main.py` to a smaller value (e.g., 10-50 replications).

## Paper Citation

If you use this code or findings in your research, please cite our paper:

```bibtex
@article{marinucci2024plsvip,
  title={Comparative Analysis of PLS-VIP and RWA Methods: A Comprehensive Simulation Study},
  author={Marinucci, Massimiliano},
  year={2024},
  url={https://github.com/m-marinucci/PLS_RWA}
}
```

## License

This project is for research and educational purposes.