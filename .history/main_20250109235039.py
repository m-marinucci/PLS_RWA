import numpy as np
import pandas as pd
from itertools import product
from factors import (
    generate_X, generate_y, get_true_importance,
    do_pls_vip, compute_rwa, top_k_accuracy
)

def format_results(results_df):
    """Format results dataframe for analysis."""
    # Convert columns to appropriate types
    results_df['n'] = results_df['n'].astype(int)
    results_df['J'] = results_df['J'].astype(int)
    results_df['magnitude'] = results_df['magnitude'].astype(str)
    results_df['noise'] = results_df['noise'].astype(str)
    results_df['rho'] = results_df['rho'].astype(float)
    results_df['method'] = results_df['method'].astype(str)
    results_df['top_k_acc'] = results_df['top_k_acc'].astype(float)
    
    return results_df

def main_fractional_simulation(
    n_reps=100,
    random_state=42,
    alpha=0.3,  # Proportion of important predictors
    randomize_importance=True  # Whether to randomize important variables
):
    """Run main simulation with dynamic importance."""
    # Set random seed for reproducibility
    rng = np.random.default_rng(random_state)
    
    # Define parameter grid
    param_grid = {
        'n': [50, 100, 200],
        'J': [7, 11, 20],
        'magnitude': ['low', 'medium', 'high'],
        'noise': ['low', 'medium', 'high'],
        'rho': [0.0, 0.5, 0.95]
    }
    
    # Create all combinations of parameters
    param_combinations = list(product(
        param_grid['n'],
        param_grid['J'],
        param_grid['magnitude'],
        param_grid['noise'],
        param_grid['rho']
    ))
    
    # Initialize results list
    results = []
    
    # Loop over parameter combinations
    for n, J, magnitude, noise, rho in param_combinations:
        print(f"\nRunning simulation for n={n}, J={J}, magnitude={magnitude}, noise={noise}, rho={rho}")
        
        # Run multiple replications
        for rep in range(n_reps):
            if rep % 10 == 0:
                print(f"  Replication {rep+1}/{n_reps}")
            
            try:
                # Generate predictors
                X = generate_X(n=n, J=J, rho=rho, random_state=rng)
                
                # Get important variables
                important_vars = get_true_importance(J, alpha=alpha, randomize=randomize_importance, rng=rng)
                
                # Generate response
                y = generate_y(X, important_vars, magnitude=magnitude, noise_label=noise, rng=rng)
                
                # Compute PLS-VIP scores
                try:
                    vip_scores = do_pls_vip(X, y)
                    if vip_scores is not None:
                        vip_acc = top_k_accuracy(vip_scores, important_vars)
                        results.append({
                            'n': n,
                            'J': J,
                            'magnitude': magnitude,
                            'noise': noise,
                            'rho': rho,
                            'method': 'PLS-VIP',
                            'top_k_acc': vip_acc,
                            'valid_run': True
                        })
                except Exception as e:
                    print(f"Error in PLS-VIP: {str(e)}")
                    results.append({
                        'n': n,
                        'J': J,
                        'magnitude': magnitude,
                        'noise': noise,
                        'rho': rho,
                        'method': 'PLS-VIP',
                        'top_k_acc': np.nan,
                        'valid_run': False
                    })
                
                # Compute RWA scores
                try:
                    rwa_scores = compute_rwa(X, y)
                    if rwa_scores is not None:
                        rwa_acc = top_k_accuracy(rwa_scores, important_vars)
                        results.append({
                            'n': n,
                            'J': J,
                            'magnitude': magnitude,
                            'noise': noise,
                            'rho': rho,
                            'method': 'RWA',
                            'top_k_acc': rwa_acc,
                            'valid_run': True
                        })
                except Exception as e:
                    print(f"Error in RWA: {str(e)}")
                    results.append({
                        'n': n,
                        'J': J,
                        'magnitude': magnitude,
                        'noise': noise,
                        'rho': rho,
                        'method': 'RWA',
                        'top_k_acc': np.nan,
                        'valid_run': False
                    })
                    
            except Exception as e:
                print(f"Error in replication {rep+1}: {str(e)}")
                continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Format results
    results_df = format_results(results_df)
    
    return results_df

def estimate_runtime(n_reps=5):
    """Estimate runtime for full simulation."""
    import time
    
    start_time = time.time()
    _ = main_fractional_simulation(n_reps=n_reps)
    end_time = time.time()
    
    # Calculate estimates
    time_per_rep = (end_time - start_time) / n_reps
    full_runtime = time_per_rep * 100  # For 100 reps
    
    print(f"\nRuntime Estimates:")
    print(f"Time per replication: {time_per_rep:.2f} seconds")
    print(f"Estimated full runtime: {full_runtime/60:.2f} minutes")
    
    return full_runtime

def analyze_performance(results_df):
    """Analyze and print performance metrics."""
    # Calculate overall performance
    overall = results_df.groupby('method')['top_k_acc'].agg(['mean', 'std', 'count'])
    print("\nOverall Performance:")
    print(overall)
    
    # Calculate performance by parameter
    for param in ['n', 'J', 'magnitude', 'noise', 'rho']:
        print(f"\nPerformance by {param}:")
        by_param = results_df.groupby(['method', param])['top_k_acc'].agg(['mean', 'std', 'count'])
        print(by_param)
    
    return overall

if __name__ == "__main__":
    # Run simulation
    results = main_fractional_simulation()
    
    # Analyze results
    overall_performance = analyze_performance(results)
    
    # Save results
    results.to_csv('simulation_results.csv', index=False)
    print("\nResults saved to simulation_results.csv")
