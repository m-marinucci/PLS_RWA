#!/usr/bin/env python3
import os
import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd, description=None):
    """Run a shell command and print its output."""
    if description:
        print(f"\n=== {description} ===")
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(1)
    return result

def setup_git():
    """Initialize git repository and create necessary branches."""
    if not Path('.git').exists():
        run_command('git init', 'Initializing git repository')
        run_command('git add .', 'Adding all files to git')
        run_command('git commit -m "Initial commit"', 'Creating initial commit')
    
    # Create and switch to fix_importance branch
    run_command('git checkout -b fix_importance', 'Creating fix_importance branch')

def setup_environment():
    """Install required packages."""
    requirements = [
        'numpy==1.24.3',  # Using older version for compatibility
        'pandas',
        'matplotlib',
        'scikit-learn',
        'statsmodels',
        'seaborn'
    ]
    
    for package in requirements:
        run_command(f'uv pip install {package}', f'Installing {package}')

def create_directories():
    """Create necessary directories for results and analysis."""
    directories = ['results', 'docs', 'figures']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def run_simulation():
    """Run the main simulation."""
    print("\n=== Running main simulation ===")
    start_time = time.time()
    run_command('uv run python main.py', 'Running main simulation')
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")

def generate_visualizations():
    """Generate all visualizations."""
    print("\n=== Generating visualizations ===")
    run_command('uv run python visualize_results.py', 'Creating visualization plots')

def perform_statistical_analysis():
    """Perform statistical analysis."""
    print("\n=== Performing statistical analysis ===")
    run_command('uv run python statistical_analysis.py', 'Running statistical analysis')

def add_comprehensive_insights():
    """Add comprehensive insights to the analysis."""
    print("\n=== Adding comprehensive insights ===")
    run_command('uv run python add_comprehensive_insights.py', 'Adding comprehensive insights')

def main():
    """Main function to reproduce the entire analysis."""
    print("Starting reproduction of analysis...")
    
    # Setup
    setup_git()
    setup_environment()
    create_directories()
    
    # Run analysis
    run_simulation()
    generate_visualizations()
    perform_statistical_analysis()
    add_comprehensive_insights()
    
    print("\nAnalysis reproduction completed successfully!")
    print("""
Results can be found in:
- results/simulation_results.csv: Raw simulation results
- figures/: Visualization plots
- docs/analysis_summary.md: Comprehensive analysis and insights
""")

if __name__ == "__main__":
    main() 