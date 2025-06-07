# PLS-VIP vs RWA Comparison Study

A statistical simulation project comparing **PLS-VIP** (Partial Least Squares - Variable Importance in Projection) and **RWA** (Relative Weight Analysis) methods for identifying important predictors in regression models.

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

## License

This project is for research and educational purposes.