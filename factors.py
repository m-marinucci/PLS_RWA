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
    if magnitude == 'Beta high':
        scale = 3.0
    elif magnitude == 'Beta medium':
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
    """Perform classical Johnson's Relative Weight Analysis (RWA).
    
    Args:
        X: Predictor matrix
        y: Response vector
        
    Returns:
        array: RWA importance scores that sum to total R^2
    """
    # 1) Standardize X and y
    Xs = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    ys = (y - y.mean()) / y.std(ddof=1)
    
    n, p = Xs.shape
    
    # 2) Correlation matrix of X (since Xs is standardized)
    R = np.corrcoef(Xs, rowvar=False)  # shape (p, p)
    
    # 3) Eigen-decomposition of R
    #    eigenvalues (evals), eigenvectors (evecs) in ascending order => reverse sort
    evals, evecs = np.linalg.eigh(R)
    idx = np.argsort(evals)[::-1]   # largest first
    evals = evals[idx]
    evecs = evecs[:, idx]           # reorder columns to match sorted evals
    
    # 4) Construct principal components Z = Xs @ evecs
    Z = Xs @ evecs  # shape (n, p), columns = principal component scores
    
    # 5) Regress ys on Z => bZ = (Z^T Z)^{-1} Z^T ys
    #    This tells us how each component predicts y
    bZ, _, _, _ = np.linalg.lstsq(Z, ys, rcond=None)  # shape (p,)
    
    # The total R^2 from this regression:
    yhat = Z @ bZ
    ss_total = np.sum(ys**2)
    ss_resid = np.sum((ys - yhat)**2)
    R2 = 1.0 - ss_resid/ss_total
    
    # 6) Compute Johnson's raw relative weights
    #    Formula: RW_j = sum_k [ bZ_k * sqrt(evals[k]) * evecs[j, k] ]^2
    RW = np.zeros(p)
    for j in range(p):
        for k in range(p):
            RW[j] += (bZ[k] * np.sqrt(evals[k]) * evecs[j, k])**2
    
    # 7) Scale so sum_j RW_j = R^2
    sum_RW = RW.sum()
    if sum_RW > 0:
        RW *= (R2 / sum_RW)
    
    return RW

###############################################################################
# 4) do_pls_vip(X, y, n_components=2)
###############################################################################
def do_pls_vip(X, y, n_components=2):
    """Compute VIP scores using PLS regression.
    
    Args:
        X (array): Predictor matrix
        y (array): Response vector
        n_components (int): Number of PLS components (minimum 2)
    
    Returns:
        array: VIP scores, one per predictor (shape: (p,))
    """
    # Ensure inputs are numpy arrays and get dimensions early
    try:
        X = np.asarray(X)
        p = X.shape[1]  # Get number of predictors early
    except:
        print("PLS-VIP error: Invalid input X")
        return np.array([0.0])  # Return single zero if we can't even get X shape
        
    try:
        y = np.asarray(y).ravel()  # Ensure y is 1D
        n = X.shape[0]
        
        # Print input dimensions for debugging
        print(f"Input shapes - X: {X.shape}, y: {y.shape}")
        
        # Ensure n_components is valid and at least 2
        n_components_orig = n_components
        n_components = min(max(n_components, 2), p, n-1)  # Clamp between 2 and min(p, n-1)
        print(f"n_components adjusted from {n_components_orig} to {n_components}")
            
        # Fit PLS
        pls = PLSRegression(n_components=n_components)
        pls.fit(X, y)
        
        # Get PLS quantities and print their shapes
        W = pls.x_weights_      # shape (p, n_components)
        T = pls.x_scores_       # shape (n, n_components)
        Q = pls.y_loadings_     # shape (1, n_components) or (n_components,)
        
        print(f"PLS matrix shapes - W: {W.shape}, T: {T.shape}, Q: {Q.shape}")
        
        # Compute fraction of Y-variance explained by each component
        SSY_component = np.zeros(n_components)
        for a in range(n_components):
            # Handle both 1D and 2D Q matrices
            q_val = float(Q[a] if Q.ndim == 1 else Q[0, a])
            comp_pred = T[:, a] * q_val
            SSY_component[a] = np.sum(comp_pred**2)
            
        total_SS = np.sum(SSY_component)
        if total_SS > 0:
            fraction_of_Y = SSY_component / total_SS
        else:
            fraction_of_Y = np.ones(n_components) / n_components
        
        # Compute VIP for each predictor
        vip_scores = np.zeros(p)
        for j in range(p):
            sum_term = 0
            for a in range(n_components):
                w_sum = np.sum(W[:, a]**2)
                if w_sum > 0:  # Avoid division by zero
                    sum_term += fraction_of_Y[a] * (W[j, a]**2 / w_sum)
            vip_scores[j] = np.sqrt(p * sum_term)
        
        return vip_scores
        
    except Exception as e:
        print(f"PLS-VIP error: {str(e)}")
        return np.zeros(p)  # Return zeros with correct dimension

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
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()
    
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
        
        # Create block diagonal correlation matrix
        R = np.zeros((J, J))
        start_idx = 0
        for block in blocks:
            size = block.shape[0]
            R[start_idx:start_idx+size, start_idx:start_idx+size] = block
            start_idx += size
    else:
        R = np.eye(J)
    
    # Generate multivariate normal X
    try:
        L = np.linalg.cholesky(R)
        X = rng.standard_normal(size=(n, J))
        X = X @ L
        
        # Standardize X
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X
    except np.linalg.LinAlgError:
        print(f"Warning: Could not generate correlated X with rho={rho}. Using uncorrelated X instead.")
        return rng.standard_normal(size=(n, J))

###############################################################################
# 7) generate_y(X, important_vars, magnitude='medium', noise_label='medium', rng=None)
###############################################################################
def generate_y(X, important_vars, magnitude='medium', noise_label='medium', rng=None):
    """Generate response variable y with linearly increasing number of important variables.
    
    Args:
        X: Predictor matrix.
        important_vars: List of indices for important variables.
        magnitude: Effect size ('low', 'medium', 'high').
        noise_label: Noise level ('low', 'medium', 'high').
        rng: Random generator for reproducibility (optional).

    Returns:
        np.ndarray: Response variable y.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, J = X.shape

    # Magnitude scaling
    magnitude_map = {
        'low': 0.5,
        'medium': 1.0,
        'high': 2.0
    }
    beta_scale = magnitude_map[magnitude]

    # Noise scaling
    noise_map = {
        'low': 0.1,
        'medium': 0.5,
        'high': 1.0
    }
    noise_scale = noise_map[noise_label]

    # Generate coefficients for important variables
    K = len(important_vars)
    beta = beta_scale * rng.uniform(0.5, 1.5, size=K)  # Randomized coefficients for important variables

    # Initialize response
    y = np.zeros(n)

    # Add effects of important variables
    for i, idx in enumerate(important_vars):
        y += beta[i] * X[:, idx]

    # Add noise
    y += rng.normal(0, noise_scale, size=n)

    # Standardize y
    y = (y - np.mean(y)) / np.std(y)
    
    return y

###############################################################################
# 8) get_true_importance(J, alpha=0.3, randomize=False, rng=None)
###############################################################################
def get_true_importance(J, alpha=0.3, randomize=False, rng=None):
    """Dynamically determine which variables are important based on J and alpha.

    Args:
        J (int): Total number of predictors.
        alpha (float): Proportion of predictors that are important.
        randomize (bool): If True, shuffle the order of important variables.
        rng: Random generator for reproducibility (optional).

    Returns:
        list: Indices of important variables.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    # Compute number of important variables (K)
    K = max(1, int(np.ceil(alpha * J)))
    
    # Select the first K variables as important
    important_vars = list(range(K))
    
    # Randomize their order if specified
    if randomize:
        rng.shuffle(important_vars)
    
    return important_vars

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

###############################################################################
# 10) discretize_data(X, y, X_points=5, y_points=7)
###############################################################################
def discretize_data(X, y, X_points=5, y_points=7):
    """Discretize continuous data into Likert scales using percentiles.
    
    Args:
        X: Predictor matrix (continuous)
        y: Response vector (continuous)
        X_points: Number of points for X Likert scale (default: 5)
        y_points: Number of points for y Likert scale (default: 7)
    
    Returns:
        tuple: (X_discrete, y_discrete)
    """
    # Convert X to Likert scale
    X_discrete = np.zeros_like(X)
    for j in range(X.shape[1]):
        # Calculate percentile bins for X
        bins = np.percentile(X[:, j], np.linspace(0, 100, X_points + 1))
        # Ensure unique bins
        bins = np.unique(bins)
        # Digitize into X_points levels (1 to X_points)
        X_discrete[:, j] = np.digitize(X[:, j], bins[:-1], right=True) + 1
    
    # Convert y to Likert scale
    # Calculate percentile bins for y
    y_bins = np.percentile(y, np.linspace(0, 100, y_points + 1))
    # Ensure unique bins
    y_bins = np.unique(y_bins)
    # Digitize into y_points levels (1 to y_points)
    y_discrete = np.digitize(y, y_bins[:-1], right=True) + 1
    
    return X_discrete, y_discrete
