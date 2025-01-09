#!/bin/bash

# Print header
echo "=== PLS-VIP vs RWA Analysis Reproduction ==="
echo "This script will reproduce all results from the analysis"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv package manager is required but not installed"
    echo "Please install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if reproduce_analysis.py exists
if [ ! -f "reproduce_analysis.py" ]; then
    echo "Error: reproduce_analysis.py not found in current directory"
    exit 1
fi

# Make sure reproduce_analysis.py is executable
chmod +x reproduce_analysis.py

echo "Starting reproduction process..."
echo "This will:"
echo "1. Set up the environment and install dependencies"
echo "2. Run the main simulation"
echo "3. Generate visualizations"
echo "4. Perform statistical analysis"
echo "5. Create comprehensive documentation"
echo

# Ask for confirmation
read -p "Do you want to proceed? [y/N] " response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Reproduction cancelled"
    exit 0
fi

echo
echo "=== Starting Reproduction ==="
# Run the reproduction script
./reproduce_analysis.py

# Check if the reproduction was successful
if [ $? -eq 0 ]; then
    echo
    echo "=== Reproduction Completed Successfully ==="
    echo
    echo "Results can be found in:"
    echo "- results/simulation_results.csv: Raw simulation results"
    echo "- figures/: Visualization plots"
    echo "- docs/analysis_summary.md: Comprehensive analysis and insights"
    echo
    echo "To view the analysis summary:"
    echo "cat docs/analysis_summary.md"
else
    echo
    echo "=== Error: Reproduction Failed ==="
    echo "Please check the error messages above"
fi 