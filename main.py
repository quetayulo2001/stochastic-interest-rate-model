"""
Stochastic Interest Rate Modelling and Prediction
Finance Club, IIT Roorkee
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Standard mathematical maturities (in years)
TENORS_YRS = np.array([0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])

def load_and_preprocess(filepath):
    """Loads, cleans, and interpolates time-series yield data."""
    if not os.path.exists(filepath):
        print(f"Error: Could not find {filepath}. Please ensure it is in the data/ directory.")
        return None
        
    df = pd.read_csv(filepath, parse_dates=True, index_col=0)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.interpolate(method='time').bfill().ffill()
    
    if df.max().max() > 1.0:
        df = df / 100.0
    return df

class CIRModel:
    """Cox-Ingersoll-Ross (CIR) Stochastic Short-Rate Model"""
    def __init__(self):
        self.kappa = None
        self.theta = None
        self.sigma = None
        
    def theoretical_yield(self, kappa, theta, sigma, r_t, tau):
        h = np.sqrt(kappa**2 + 2 * sigma**2)
        exp_ht = np.exp(h * tau)
        
        B = (2 * (exp_ht - 1)) / (2 * h + (kappa + h) * (exp_ht - 1))
        
        term1 = 2 * h * np.exp((kappa + h) * tau / 2)
        term2 = 2 * h + (kappa + h) * (exp_ht - 1)
        A = (term1 / term2) ** ((2 * kappa * theta) / sigma**2)
        
        return (B * r_t - np.log(A)) / tau

    def panel_objective(self, params, train_data, tenors):
        kappa, theta, sigma = params
        if kappa <= 0 or theta <= 0 or sigma <= 0 or (2 * kappa * theta < sigma**2):
            return 1e9
            
        r_t_array = train_data.iloc[:, 0].values 
        actual_yields = train_data.values
        
        preds = np.array([self.theoretical_yield(kappa, theta, sigma, r, tenors) for r in r_t_array])
        return np.mean((preds - actual_yields)**2)

    def fit_global(self, train_data, tenors):
        bounds = [(1e-4, 3.0), (1e-4, 0.5), (1e-4, 0.5)]
        print("Calibrating Global CIR Model... (This may take 10-15 seconds)")
        
        result = differential_evolution(
            self.panel_objective, bounds, args=(train_data, tenors),
            strategy='best1bin', popsize=15, tol=1e-6
        )
        
        self.kappa, self.theta, self.sigma = result.x
        feller = 2 * self.kappa * self.theta - self.sigma**2
        
        print(f"\n--- Global Calibration Results ---")
        print(f"Kappa (Speed of Mean Reversion): {self.kappa:.4f}")
        print(f"Theta (Long-Run Asymptotic Mean): {self.theta:.4f}")
        print(f"Sigma (Volatility Parameter): {self.sigma:.4f}")
        print(f"Feller Condition Check: {feller:.6f} (>0 is VALID)")

def predict_and_evaluate(model, test_data, test_3m, tenors, use_extension=False, train_data=None):
    """Predicts out-of-sample yields using Base or Transient CIR++ methodology."""
    actual_yields = test_data.values
    r_t_series = test_3m.values.flatten()
    predicted_yields = []
    
    # Dynamic Alignment for Evaluation Columns
    if train_data is not None and test_data.shape[1] != len(tenors):
        try:
            col_indices = [train_data.columns.get_loc(col) for col in test_data.columns if col in train_data.columns]
        except KeyError:
            col_indices = list(range(len(tenors) - test_data.shape[1], len(tenors)))
    else:
        col_indices = list(range(len(tenors)))
        
    initial_shift = np.zeros(len(tenors))
    if use_extension and train_data is not None:
        last_train_rt = train_data.iloc[-1, 0]
        last_train_actuals = train_data.iloc[-1].values
        last_train_base_pred = model.theoretical_yield(model.kappa, model.theta, model.sigma, last_train_rt, tenors)
        initial_shift = last_train_actuals - last_train_base_pred

    decay_gamma = 0.05 
            
    for i, r_t in enumerate(r_t_series):
        base_pred = model.theoretical_yield(model.kappa, model.theta, model.sigma, r_t, tenors)
        
        if use_extension:
            time_decay_weight = np.exp(-decay_gamma * i)
            adjusted_pred = base_pred + (initial_shift * time_decay_weight)
        else:
            adjusted_pred = base_pred
            
        predicted_yields.append(adjusted_pred[col_indices])
        
    predicted_yields = np.array(predicted_yields)
    r2 = r2_score(actual_yields, predicted_yields)
    return predicted_yields, r2, col_indices

if __name__ == "__main__":
    print("Loading Data Pipeline...")
    train_df = load_and_preprocess('data/train_data.csv')
    test_df = load_and_preprocess('data/test_data.csv')
    test_3m_df = load_and_preprocess('data/test_data_3M.csv')
    
    if train_df is not None and test_df is not None and test_3m_df is not None:
        # Instantiate and Calibrate
        cir = CIRModel()
        cir.fit_global(train_df, TENORS_YRS)
        
        # Evaluate
        print("\n--- Out-of-Sample Evaluation ---")
        base_preds, base_r2, col_idx = predict_and_evaluate(cir, test_df, test_3m_df, TENORS_YRS, use_extension=False, train_data=train_df)
        print(f"Base CIR Out-of-Sample R^2: {base_r2:.4f}")
        
        ext_preds, ext_r2, _ = predict_and_evaluate(cir, test_df, test_3m_df, TENORS_YRS, use_extension=True, train_data=train_df)
        print(f"Transient CIR++ Out-of-Sample R^2: {ext_r2:.4f}")
        
        if ext_r2 > 0.85 and base_r2 > 0.85:
            print("\nSUCCESS: Strict >0.85 evaluation criteria met.")
                
        # Plotting
        plot_tenors = TENORS_YRS[col_idx]
        day_idx = len(test_df) // 2
        
        plt.plot(plot_tenors, test_df.iloc[day_idx].values, label='Actual Yield Curve', marker='o', color='black', linewidth=2)
        plt.plot(plot_tenors, base_preds[day_idx], label='Base CIR Formulation', linestyle='--', color='gray')
        plt.plot(plot_tenors, ext_preds[day_idx], label='Transient CIR++ Extension', linestyle='-.', color='crimson', linewidth=2)
        plt.title(f"Out-of-Sample Yield Curve Reconstruction (Test Day {day_idx})")
        plt.xlabel("Maturity Tenors (Years)")
        plt.ylabel("Yield (Decimal)")
        plt.legend()
        plt.savefig('reconstruction_plot.png')
        print("\nPlot saved successfully as 'reconstruction_plot.png'.")