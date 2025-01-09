import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from factors import generate_X, generate_y, get_true_importance, compute_rwa, do_pls_vip, compute_top_k_accuracy
from logger import setup_logger
import time
from datetime import timedelta
from tqdm import tqdm  # type: ignore
import logging
from statsmodels.formula.api import ols  # type: ignore
from scipy import stats  # type: ignore

def format_results(results_df):
    """Format results in a concise way."""
    summary = {}
    
    # Overall performance
    for method in ['rwa', 'pls']:
        summary[method] = {}
        for k in [1, 2, 3]:
            col = f'top{k}_{method}'
            mean = results_df[col].mean()
            std = results_df[col].std()
            summary[method][k] = f"{mean:.3f} (±{std:.3f})"
    
    return summary

def main_fractional_simulation(n_rep=1000, debug_mode=False):
    """Run main fractional factorial simulation."""
    logger = setup_logger(debug_mode)
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    # Design parameters
    n_values = [50, 275, 500]  # Sample sizes
    J_values = [7, 11, ]  # Number of predictors
    magnitude_values = ['low', 'medium', 'high']  # Effect magnitudes
    noise_values = ['low', 'medium', 'high']  # Noise levels
    rho_values = [0.0, 0.5, 0.95]  # Correlation levels
    
    # Calculate total combinations
    total_combinations = len(n_values) * len(J_values) * len(magnitude_values) * len(noise_values) * len(rho_values)
    
    # Initialize results DataFrame
    results = []
    
    # Create overall progress bar
    logger.info("Starting simulation experiments...")
    progress_bar = tqdm(total=total_combinations, desc="Overall Progress", position=0)
    
    start_time = time.time()
    
    # Run simulation for each combination
    for n in n_values:
        for J in J_values:
            for magnitude in magnitude_values:
                for noise in noise_values:
                    for rho in rho_values:
                        if debug_mode:
                            logger.debug(f"\nSettings: n={n}, J={J}, magnitude={magnitude}, noise={noise}, rho={rho}")
                        
                        valid_runs = 0
                        top1_rwa = top2_rwa = top3_rwa = 0
                        top1_pls = top2_pls = top3_pls = 0
                        
                        # Run replications
                        for rep in range(n_rep):
                            try:
                                X = generate_X(n, J, rho=rho, random_state=rep)
                                y = generate_y(X, magnitude=magnitude, noise_label=noise, random_state=rep)
                                true_importance = get_true_importance(J)
                                
                                rwa_importance = compute_rwa(X, y)
                                pls_importance = do_pls_vip(X, y)
                                
                                valid_runs += 1
                                top1_rwa += compute_top_k_accuracy(true_importance, rwa_importance, k=1)
                                top2_rwa += compute_top_k_accuracy(true_importance, rwa_importance, k=2)
                                top3_rwa += compute_top_k_accuracy(true_importance, rwa_importance, k=3)
                                top1_pls += compute_top_k_accuracy(true_importance, pls_importance, k=1)
                                top2_pls += compute_top_k_accuracy(true_importance, pls_importance, k=2)
                                top3_pls += compute_top_k_accuracy(true_importance, pls_importance, k=3)
                                
                            except Exception as e:
                                if debug_mode:
                                    logger.debug(f"Error in run {rep}: {str(e)}")
                                continue
                        
                        # Store results if we have valid runs
                        if valid_runs > 0:
                            results.append({
                                'n': n,
                                'J': J,
                                'magnitude': magnitude,
                                'noise_label': noise,
                                'rho': rho,
                                'valid_runs': valid_runs,
                                'top1_rwa': top1_rwa / valid_runs,
                                'top2_rwa': top2_rwa / valid_runs,
                                'top3_rwa': top3_rwa / valid_runs,
                                'top1_pls': top1_pls / valid_runs,
                                'top2_pls': top2_pls / valid_runs,
                                'top3_pls': top3_pls / valid_runs
                            })
                        
                        progress_bar.update(1)
    
    end_time = time.time()
    elapsed_time = timedelta(seconds=int(end_time - start_time))
    
    progress_bar.close()
    logger.info(f"\nSimulation completed in {elapsed_time}!")
    
    return pd.DataFrame(results)

def estimate_runtime(n_rep_sample=10):
    """Estimate the runtime for the full simulation based on a small sample.
    
    Args:
        n_rep_sample (int): Number of replications to use for estimation
        
    Returns:
        tuple: (estimated_seconds, sample_results_df)
    """
    start_time = time.time()
    df_sample = main_fractional_simulation(n_rep=n_rep_sample)
    sample_runtime = time.time() - start_time
    
    # Estimate full runtime
    estimated_seconds = (sample_runtime / n_rep_sample) * 1000
    
    # Print runtime info
    print("\nRuntime Estimation:")
    print(f"Sample runtime: {timedelta(seconds=int(sample_runtime))}")
    print(f"Success rate: {df_sample['valid_runs'].mean() / n_rep_sample * 100:.1f}%")
    print(f"Estimated full runtime: {timedelta(seconds=int(estimated_seconds))}")
    
    return estimated_seconds, df_sample

def analyze_performance(df, method_name, k_values=[1, 2, 3]):
    """Analyze performance metrics for a given method.
    
    Args:
        df: DataFrame with results
        method_name: 'rwa' or 'pls'
        k_values: list of k values to analyze
    """
    print(f"\nAnalysis of {method_name.upper()} performance:")
    
    # Analyze each k value
    for k in k_values:
        col_name = f'top{k}_{method_name}'
        
        print(f"\nTop-{k} Accuracy Summary:")
        print(df[col_name].describe())
        
        # Fit regression model
        from statsmodels.formula.api import ols
        model = ols(f"{col_name} ~ C(n) + C(J) + C(magnitude) + C(noise_label) + C(rho)", 
                   data=df).fit()
        print(f"\nRegression Analysis for Top-{k} Accuracy:")
        print(model.summary().tables[1])  # Print only coefficients table
        
        # Performance by factor levels
        factors = ['n', 'J', 'magnitude', 'noise_label', 'rho']
        print("\nMean Accuracy by Factor Levels:")
        for factor in factors:
            means = df.groupby(factor)[col_name].mean()
            std = df.groupby(factor)[col_name].std()
            print(f"\n{factor}:")
            for level in means.index:
                print(f"  {level}: {means[level]:.3f} (±{std[level]:.3f})")

if __name__ == "__main__":
    # First estimate runtime
    estimated_seconds, df_sample = estimate_runtime(n_rep_sample=10)
    
    # Show sample results
    print("\nSample results (first few rows):")
    print(df_sample.head())
    print(f"\nProceeding with full simulation (estimated {timedelta(seconds=int(estimated_seconds))})...")
    
    # Run full simulation
    print("\nRunning full simulation...")
    df_results = main_fractional_simulation(n_rep=1000)
    print("\nFirst 10 rows of results:")
    print(df_results.head(10))
    print("\nSummary of valid runs:")
    print(df_results['valid_runs'].describe())
    
    # Remove rows with no valid runs for analysis
    df_analysis = df_results[df_results['valid_runs'] > 0].copy()
    
    if len(df_analysis) > 0:
        # Analyze RWA performance
        analyze_performance(df_analysis, 'rwa')
        
        # Analyze PLS performance
        analyze_performance(df_analysis, 'pls')
        
        # Compare RWA vs PLS
        print("\nComparison of RWA vs PLS:")
        for k in [1, 2, 3]:
            rwa_col = f'top{k}_rwa'
            pls_col = f'top{k}_pls'
            
            print(f"\nTop-{k} Accuracy Comparison:")
            rwa_mean = df_analysis[rwa_col].mean()
            rwa_std = df_analysis[rwa_col].std()
            pls_mean = df_analysis[pls_col].mean()
            pls_std = df_analysis[pls_col].std()
            
            print(f"RWA: {rwa_mean:.3f} (±{rwa_std:.3f})")
            print(f"PLS: {pls_mean:.3f} (±{pls_std:.3f})")
            
            # Paired t-test
            from scipy import stats
            t_stat, p_val = stats.ttest_rel(df_analysis[rwa_col], df_analysis[pls_col])
            print(f"Paired t-test: t={t_stat:.3f}, p={p_val:.3f}")
        
        # Save detailed results
        df_analysis.to_csv("detailed_results.csv", index=False)
        
        # Create visualizations
        from visualization import plot_comparison
        plot_comparison(df_analysis, save_path='performance_comparison.png')

    # Run simulation
    results = main_fractional_simulation(n_rep=1000, debug_mode=False)
    
    # Save results
    results.to_csv("simulation_results.csv", index=False)
    print("\nResults saved to simulation_results.csv")
    
    # Show concise summary
    summary = format_results(results)
    print("\nOverall Performance Summary:")
    print("\nRWA Performance:")
    for k in [1, 2, 3]:
        print(f"Top-{k}: {summary['rwa'][k]}")
    print("\nPLS Performance:")
    for k in [1, 2, 3]:
        print(f"Top-{k}: {summary['pls'][k]}")
    
    # Create visualizations
    try:
        from visualization import plot_comparison
        plot_comparison(results)
        print("\nVisualization saved as 'comparison_plot.png'")
    except ImportError as e:
        print(f"\nVisualization skipped: {str(e)}")
