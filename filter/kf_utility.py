import numpy as np
import pandas as pd
from scipy.linalg import expm, eig, eigh

def discretize_lti_system(F, L, Qc, dt):
    """
    Discretize continuous time state space matrices.
    Inputs: continuous state transition matrix (F), process noise matrix (L), Continuous time noise (Qc),
            time difference with respect to previous obesrvation (dt)
    outputs: discrete state transition matrix (Fd), discrete process noise matrix (Qd)
    
    Discretization method follows the method described by van Loan (1978)
    C. F. Van Loan, “Computing Integrals Involving the Matrix Exponential,” 
    IEEE Transactions on Automatic Control, vol. 23, no. 3, pp. 395–404, 1978.
    """
    if np.isscalar(Qc):
        n = F.shape[0]
        M = np.block([
            [ -F,            Qc * (L @ L.T)],
            [ np.zeros((n,n)),  F.T ]
        ])
    if isinstance(Qc, np.ndarray):
        n = F.shape[0]
        M = np.block([
            [ -F,            L @ Qc @ L.T],
            [ np.zeros((n,n)),  F.T ]
        ])
    
    M_exp = expm(M * dt)
    
    B = M_exp[:n,  n:2*n]  
    C = M_exp[n:2*n, n:2*n]

    F_d = C.T
    Q_d = F_d @ B
    Q_d = 0.5 * (Q_d + Q_d.T)
    return F_d, Q_d

def is_stable_continuous(Fc):
    eigvals, _ = eig(Fc)
    if np.any(np.real(eigvals) > 0):
        raise ValueError(f"Matrix is not stable: {eigvals}")
    
def is_stable_discrete(Fd):
    eigvals, _ = eig(Fd)
    if np.any(np.abs(eigvals) > 1):
        raise ValueError(f"matrix is not stable: {eigvals}")

def ensure_psd(matrix, epsilon=1e-6):
    matrix = (matrix + matrix.T) / 2
    eigvals = eigh(matrix, eigvals_only=True, check_finite=False)
    min_eig = np.min(eigvals)
    if min_eig < 0:
        correction = abs(min_eig) + epsilon  # Ensure positive definiteness
        matrix += np.eye(matrix.shape[0]) * correction
    return matrix

def calculate_forecast_errors(predictions, actuals, metrics=['mae', 'rmse', 'mse', 'da'], return_full=True):
    predictions = pd.Series(predictions, name='prediction')
    actuals = pd.Series(actuals, name='actual')

    if len(predictions) != len(actuals):
        raise ValueError("Length of predictions and actuals must be the same.")

    df = pd.DataFrame({'prediction': predictions, 'actual': actuals})
    epsilon = 1e-10
    if 'mae' in metrics or 'mase' in metrics:
        df['abs_error'] = (df['actual'] - df['prediction']).abs()
    if 'mse' in metrics or 'rmse' in metrics:
        df['squared_error'] = (df['actual'] - df['prediction']) ** 2
    if 'mape' in metrics:
        df['mape'] = (df['abs_error'] / (df['actual'].abs() + epsilon)) * 100
    if 'smape' in metrics:
        df['smape'] = (df['abs_error'] / ((df['actual'].abs() + df['prediction'].abs()) / 2 + epsilon)) * 100
    if 'da' in metrics:
        df['direction_correct'] = (np.sign(df['actual']) == np.sign(df['prediction'])).astype(int)

    summary = {}
    if 'mae' in metrics:
        summary['mae'] = df['abs_error'].mean()
    if 'mse' in metrics:
        summary['mse'] = df['squared_error'].mean()
    if 'rmse' in metrics:
        summary['rmse'] = np.sqrt(df['squared_error'].mean())
    if 'mape' in metrics:
        summary['mape'] = df['mape'].mean()
    if 'smape' in metrics:
        summary['smape'] = df['smape'].mean()
    if 'da' in metrics:
        summary['da'] = df['direction_correct'].mean() * 100
    if 'mase' in metrics:
        naive_mae = np.mean(np.abs(df['actual'].values[1:] - df['actual'].values[:-1]))
        if naive_mae == 0:
            raise ValueError("Cannot compute MASE because the naive forecast has zero MAE.")
        summary['mase'] = df['abs_error'].mean() / naive_mae
    if return_full:
        return df, pd.DataFrame([summary])
    else:
        return pd.DataFrame([summary])
