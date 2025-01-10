# PLS-RWA Comparative Analysis

## Overview

This repository contains a comprehensive analysis framework comparing Partial Least Squares Variable Importance in Projection (PLS-VIP) and Relative Weight Analysis (RWA) methods. The codebase implements simulation studies to evaluate these methods across various conditions and data characteristics.

## Core Components

### Analysis Scripts

- `main.py` - Primary simulation engine and analysis orchestrator
- `factors.py` - Implementation of experimental factors and condition generation
- `reproduce_analysis.py` - Script to reproduce all analysis results
- `statistical_analysis.py` - Statistical analysis of simulation results
- `visualize_results.py` - Generation of visualization outputs
- `generate_summary.py` - Creation of summary statistics and tables

### Data Processing

- `add_insights.py` - Basic insight generation from results
- `add_comprehensive_insights.py` - Detailed analysis and pattern extraction

### Reproduction

- `reproduce_results.sh` - Shell script to reproduce the entire analysis pipeline
- Key output files:
  - `simulation_results.csv` - Raw simulation results
  - `statistical_analysis_results.txt` - Statistical analysis outputs
  - Various visualization outputs in `docs/figures/`

## Getting Started

1. Clone this repository

```bash
git clone https://github.com/m-marinucci/PLS_RWA.git
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the analysis

```bash
./reproduce_results.sh
```

## Project Structure

```
.
├── *.py                # Core Python analysis scripts
├── docs/
│   ├── figures/       # Generated visualizations
│   └── tables/        # Analysis result tables
└── paper/             # Research paper materials (supplementary)
```

## Results

The analysis results are organized in:

- Raw data: `simulation_results.csv`
- Statistical summaries: `docs/tables/`
- Visualizations: `docs/figures/`
- Detailed findings: `docs/analysis_summary.md`

## License

Proprietary

## Contact

Massimiliano Marinucci (<mmarinucci@numinate.com>)
