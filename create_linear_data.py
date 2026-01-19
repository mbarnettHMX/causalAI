import pandas as pd
import numpy as np

def create_linear_dataset():
    """Create a linear dataset similar to DoWhy's datasets.linear_dataset()"""
    n = 1000
    np.random.seed(42)  # For reproducibility
    
    # Create instrumental variables
    Z0 = np.random.normal(0, 1, n)
    Z1 = np.random.normal(0, 1, n)
    
    # Create confounders
    W0 = np.random.normal(0, 1, n)
    W1 = np.random.normal(0, 1, n) 
    W2 = np.random.normal(0, 1, n)
    W3 = np.random.normal(0, 1, n)
    W4 = np.random.normal(0, 1, n)
    
    # Create effect modifiers
    X0 = np.random.normal(0, 1, n)
    X1 = np.random.normal(0, 1, n)
    
    # Create treatment (affected by instruments and confounders)
    v0_noise = np.random.normal(0, 0.1, n)
    v0 = 0.5 * Z0 + 0.3 * Z1 + 0.4 * W0 + 0.3 * W1 + 0.2 * W2 + v0_noise
    v0 = (v0 > np.median(v0)).astype(int)  # Binary treatment
    
    # Create front-door variable (mediator)
    FD0_noise = np.random.normal(0, 0.1, n)
    FD0 = 0.7 * v0 + FD0_noise
    
    # Create outcome
    y_noise = np.random.normal(0, 0.1, n)
    y = 0.6 * FD0 + 0.3 * W0 + 0.2 * W1 + 0.15 * W2 + 0.4 * W3 + 0.3 * W4 + 0.2 * X0 + 0.15 * X1 + y_noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'Z0': Z0, 'Z1': Z1,
        'W0': W0, 'W1': W1, 'W2': W2, 'W3': W3, 'W4': W4,
        'X0': X0, 'X1': X1,
        'v0': v0,
        'FD0': FD0,
        'y': y
    })
    
    return data

if __name__ == "__main__":
    data = create_linear_dataset()
    print(f"Dataset created with shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"First few rows:")
    print(data.head(3))