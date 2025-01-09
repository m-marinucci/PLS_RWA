import numpy as np
from sklearn.cross_decomposition import PLSRegression  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

###############################################################################
# 1) generate_true_betas(J, magnitude, rng)
###############################################################################
def generate_true_betas(J, magnitude, rng):
    """Generate sorted betas for the chosen magnitude.
    
    Args:
        J (int): Number of predictors
        magnitude (str): 'low', 'medium', or 'high'
        rng: numpy random number generator
    
    Returns:
        numpy array: Sorted beta coefficients
    """
    if magnitude == 'high':
        scale = 3.0
    elif magnitude == 'medium':
        scale = 1.5
    else:  # low
        scale = 0.5
    
    raw = rng.uniform(low=0.2, high=1.0, size=J) * scale
    return -np.sort(-raw)  # Sort in descending order

###############################################################################
# 2) simulate_one_rep(n, J, beta, noise_var, rho, rng)
###############################################################################
def simulate_one_rep(n, J, beta, noise_var, rho, rng):
    """Generate one replication of data.
    
    Args:
        n (int): Sample size
        J (int): Number of predictors
        beta (array): True coefficients
        noise_var (float): Noise variance
        rho (float): Correlation between predictors
        rng: numpy random number generator
    
    Returns:
        tuple: (X, y) or (None, None) if correlation matrix is invalid
    """
    if not (0 <= rho < 1):
        return None, None
    
    # Create correlation matrix
    R = np.full((J, J), rho)
    np.fill_diagonal(R, 1.0)
    
    try:
        # Generate multivariate normal X
        L = np.linalg.cholesky(R)
        X = rng.standard_normal(size=(n, J))
        X = X @ L
        
        # Standardize X
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Generate y with specified noise variance
        epsilon = np.sqrt(noise_var) * rng.standard_normal(size=n)
        y = X @ beta + epsilon
        
        return X, y
    except np.linalg.LinAlgError:
        return None, None

###############################################################################
# 3) compute_rwa(X, y)
###############################################################################
def compute_rwa(X, y):
    """Compute RWA importance scores for predictors.
    
    Args:
        X: Predictor matrix
        y: Response vector
        
    Returns:
        array: RWA importance scores
    """
    # Standardize X and y
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
    
    # Fit OLS model
    n = X.shape[0]
    X_with_intercept = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    
    # Extract coefficients (excluding intercept)
    beta = beta[1:]
    
    # Compute RWA scores (absolute standardized coefficients)
    rwa_scores = np.abs(beta)
    
    return rwa_scores

###############################################################################
# 4) do_pls_vip(X, y, n_components=2)
###############################################################################
def do_pls_vip(X, y, n_components=2):
    """Compute VIP scores using PLS regression.
    
    Args:
        X (array): Predictor matrix
        y (array): Response vector
        n_components (int): Number of PLS components
    
    Returns:
        array: VIP scores
    """
    try:
        # Ensure inputs are numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Fit PLS
        n_components = min(n_components, X.shape[1], X.shape[0])
        pls = PLSRegression(n_components=n_components)
        pls.fit(X, y.reshape(-1, 1))
        
        # Get weights and loadings
        w = pls.x_weights_
        q = pls.y_loadings_
        t = pls.x_scores_
        p = X.shape[1]
        
        # Calculate VIP scores
        m = np.square(w).sum(axis=0)
        weighted = np.square(t).sum(axis=0) * np.square(q)
        vip = np.sqrt(p * np.sum(m * weighted.reshape(-1,1), axis=1) / np.sum(weighted))
        
        return vip
    except Exception as e:
        print(f"PLS-VIP error: {str(e)}")
        return None

###############################################################################
# 5) top_k_accuracy(est_importance, true_betas, k=1)
###############################################################################
def top_k_accuracy(est_importance, true_betas, k=1):
    """Compute fraction overlap of top-k predicted vs. top-k true.
    
    Args:
        est_importance (array): Estimated importance scores
        true_betas (array): True beta coefficients
        k (int): Number of top items to consider
    
    Returns:
        float: Fraction of overlap between top-k sets
    """
    if est_importance is None or true_betas is None:
        return 0.0
    
    # Convert to numpy arrays if not already
    est_importance = np.asarray(est_importance)
    true_betas = np.asarray(true_betas)
    
    # Get indices of top k items as numpy arrays
    true_top_k = np.argsort(-np.abs(true_betas))[:k]
    est_top_k = np.argsort(-np.abs(est_importance))[:k]
    
    # Compute overlap using numpy operations
    overlap = np.sum(np.isin(true_top_k, est_top_k))
    return float(overlap) / k

###############################################################################
# 6) generate_X(n, J, rho=0.0, random_state=None)
###############################################################################
def generate_X(n, J, rho=0.0, random_state=None):
    """Generate predictor matrix X with correlation structure.
    
    Args:
        n: Number of samples
        J: Number of predictors
        rho: Base correlation between predictors
        random_state: Random state for reproducibility
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate correlation matrix with block structure
    if rho > 0:
        # Create blocks of correlated variables
        block_size = max(2, J // 3)
        num_blocks = J // block_size
        remainder = J % block_size
        
        blocks = []
        for i in range(num_blocks):
            block_corr = np.full((block_size, block_size), rho)
            np.fill_diagonal(block_corr, 1.0)
            blocks.append(block_corr)
        
        if remainder > 0:
            block_corr = np.full((remainder, remainder), rho)
            np.fill_diagonal(block_corr, 1.0)
            blocks.append(block_corr)
        
        corr = np.zeros((J, J))
        start_idx = 0
        for block in blocks:
            size = block.shape[0]
            corr[start_idx:start_idx+size, start_idx:start_idx+size] = block
            start_idx += size
        
        # Add some random correlation between blocks
        mask = corr == 0
        corr[mask] = np.random.uniform(-0.2, 0.2, size=mask.sum())
        np.fill_diagonal(corr, 1.0)
        
        # Ensure correlation matrix is positive definite
        eigvals = np.linalg.eigvals(corr)
        if np.min(eigvals) < 0:
            corr += np.eye(J) * (abs(np.min(eigvals)) + 0.01)
        
        # Generate correlated data
        L = np.linalg.cholesky(corr)
        X = np.random.standard_normal((n, J)) @ L.T
    else:
        X = np.random.standard_normal((n, J))
    
    return X

###############################################################################
# 7) generate_y(X, magnitude='medium', noise_label='medium', random_state=None)
###############################################################################
def generate_y(X, magnitude='medium', noise_label='medium', random_state=None):
    """Generate response variable y with non-linear effects and interactions.
    
    Args:
        X: Predictor matrix
        magnitude: Effect size ('low', 'medium', 'high')
        noise_label: Noise level ('low', 'medium', 'high')
        random_state: Random state for reproducibility
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n, J = X.shape
    
    # Set magnitude of effects
    magnitude_map = {
        'low': 0.5,
        'medium': 1.0,
        'high': 2.0
    }
    beta_scale = magnitude_map[magnitude]
    
    # Set noise level
    noise_map = {
        'low': 0.1,
        'medium': 0.5,
        'high': 1.0
    }
    noise_scale = noise_map[noise_label]
    
    # Generate true effects based on available variables
    y = np.zeros(n)
    
    # Always include first variable if available
    if J >= 1:
        y += beta_scale * X[:, 0]  # Linear effect
    
    # Add quadratic effect of second variable if available
    if J >= 2:
        y += beta_scale * 0.5 * X[:, 1]**2
    
    # Add interaction effect if we have at least 3 variables
    if J >= 3:
        y += beta_scale * 0.3 * X[:, 0] * X[:, 2]
    
    # Add noise
    y += np.random.normal(0, noise_scale, n)
    
    # Standardize response
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
    
    return y

###############################################################################
# 8) get_true_importance(J)
###############################################################################
def get_true_importance(J):
    """Return indices of truly important variables.
    Always returns [0] for J=1, [0,1] for J=2, and [0,1,2] for J>=3."""
    return [0, 1, 2][:J]  # First three variables are important (will be truncated as needed)

###############################################################################
# 9) compute_top_k_accuracy(true_importance, est_importance, k=1)
###############################################################################
def compute_top_k_accuracy(true_importance, est_importance, k=1):
    """Compute top-k accuracy between true and estimated importance rankings.
    
    Args:
        true_importance: List of indices of truly important variables
        est_importance: Array of importance scores
        k: Number of top variables to consider
        
    Returns:
        float: Proportion of overlap between top-k variables
    """
    # Ensure k is not larger than the number of variables
    k = min(k, len(est_importance))
    k = min(k, len(true_importance))
    
    # Get top k indices from estimated importance
    top_k_indices = np.argsort(-est_importance)[:k]
    
    # Get top k indices from true importance (in case they're not already sorted)
    true_top_k = true_importance[:k]
    
    # Compute overlap
    overlap = len(set(true_top_k) & set(top_k_indices))
    return overlap / k
