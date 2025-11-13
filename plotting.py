import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def _resolve_window_length(series_length, requested=None):
    """Resolve a valid moving-average window for the requested length."""
    if series_length < 3:
        return None
    if requested is None:
        auto_window = max(3, min(max(series_length // 10, 3), 40))
        return min(auto_window, series_length)
    try:
        requested = int(requested)
    except Exception:
        requested = 3
    requested = max(3, requested)
    return min(requested, series_length)

def create_feature_plots(dfkep, encoded_target, thresholded, min_data_points=None, window_size=None):
    """
    Generate correlation plots for all top features.
    
    Parameters:
    -----------
    dfkep : pandas.DataFrame
        The main dataset
    encoded_target : pandas.Series
        The encoded target variable
    thresholded : list
        List of (feature_name, correlation_value) tuples for top features
    min_data_points : int
        Minimum number of non-null data points required (ignored if None)
    window_size : int | None
        Requested moving-average window size; falls back to adaptive sizing when None
    
    Returns:
    --------
    None (prints FEATURE_PLOT lines to stdout)
    """
    plot_dir = os.path.join('static', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    for feat_name, corr_val in thresholded:
        try:
            x_series = dfkep[feat_name]
            mask = x_series.notna()
            x_vals = x_series[mask].to_numpy()
            y_vals = pd.Series(encoded_target)[mask].to_numpy()
            unique_vals = np.unique(x_vals)
            # Check unique values requirement (skip binary features)
            if len(unique_vals) < 3:
                print(f"Skipping plot for {feat_name}: only {len(unique_vals)} unique values (minimum 3 required)")
                continue
            
            if len(x_vals) > 1:
                order_idx = x_vals.argsort()
                x_sorted = x_vals[order_idx]
                y_sorted = y_vals[order_idx]
                
                plt.figure(figsize=(6, 3))
                plt.plot(x_sorted, y_sorted, color='#6ea8fe', linewidth=1.5)
                plt.scatter(x_sorted, y_sorted, s=6, color='#6ea8fe', alpha=0.6)
                
                resolved_window = _resolve_window_length(len(y_sorted), window_size)
                if resolved_window:
                    moving_avg = np.convolve(y_sorted, np.ones(resolved_window)/resolved_window, mode='valid')
                    x_ma = x_sorted[resolved_window-1:]
                    plt.plot(
                        x_ma,
                        moving_avg,
                        color='red',
                        linewidth=2,
                        alpha=0.8,
                        label=f'{resolved_window}-point Moving Average'
                    )
                    plt.legend(fontsize=8, loc='upper right')
                
                plt.title(f"{feat_name} vs Encoded Target", fontsize=10)
                plt.xlabel(feat_name, fontsize=9)
                plt.ylabel("Encoded Target", fontsize=9)
                plt.grid(True, alpha=0.2)
                plt.tight_layout()
                
                # Save with feature name in filename
                safe_name = feat_name.replace('/', '_').replace('\\', '_')
                out_path = os.path.join(plot_dir, f"kepler_feature_{safe_name}.png")
                plt.savefig(out_path, dpi=120)
                plt.close()
                print(f"FEATURE_PLOT: kepler.csv -> {feat_name} -> {out_path}")
        except Exception as e:
            print(f"Error creating plot for {feat_name}: {e}")


def create_annotated_feature_plot(dfkep, encoded_target, feat_name, user_value, predicted_value, window_size=None):
    """
    Generate a correlation plot for a specific feature with user's input point highlighted.
    
    Parameters:
    -----------
    dfkep : pandas.DataFrame
        The main dataset
    encoded_target : pandas.Series
        The encoded target variable
    feat_name : str
        The feature name to plot
    user_value : float
        The user's input value for this feature
    predicted_value : float
        The predicted encoded target value for this feature
    
    window_size : int | None
        Requested moving-average window size; falls back to adaptive sizing when None

    Returns:
    --------
    str : Path to the generated plot
    """
    plot_dir = os.path.join('static', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    try:
        x_series = dfkep[feat_name]
        mask = x_series.notna()
        x_vals = x_series[mask].to_numpy()
        y_vals = pd.Series(encoded_target)[mask].to_numpy()
        
        if len(x_vals) > 1:
            order_idx = x_vals.argsort()
            x_sorted = x_vals[order_idx]
            y_sorted = y_vals[order_idx]
            
            plt.figure(figsize=(6, 3))
            plt.plot(x_sorted, y_sorted, color='#6ea8fe', linewidth=1.5, alpha=0.5)
            plt.scatter(x_sorted, y_sorted, s=6, color='#6ea8fe', alpha=0.3)
            
            resolved_window = _resolve_window_length(len(y_sorted), window_size)
            if resolved_window:
                moving_avg = np.convolve(y_sorted, np.ones(resolved_window)/resolved_window, mode='valid')
                x_ma = x_sorted[resolved_window-1:]
                plt.plot(
                    x_ma,
                    moving_avg,
                    color='red',
                    linewidth=2,
                    alpha=0.8,
                    label=f'{resolved_window}-point Moving Average'
                )
            
            # Plot user's point
            plt.scatter([user_value], [predicted_value], s=100, color='#00ff00', marker='*', 
                       edgecolors='white', linewidths=1.5, zorder=5, label='Your Input')
            
            plt.title(f"{feat_name} vs Encoded Target", fontsize=10)
            plt.xlabel(feat_name, fontsize=9)
            plt.ylabel("Encoded Target", fontsize=9)
            plt.grid(True, alpha=0.2)
            plt.legend(fontsize=8, loc='upper right')
            plt.tight_layout()
            
            # Save with feature name and timestamp in filename
            safe_name = feat_name.replace('/', '_').replace('\\', '_')
            out_path = os.path.join(plot_dir, f"kepler_feature_{safe_name}_annotated.png")
            plt.savefig(out_path, dpi=120)
            plt.close()
            return out_path
        return None
    except Exception as e:
        print(f"Error creating annotated plot for {feat_name}: {e}")
        return None

