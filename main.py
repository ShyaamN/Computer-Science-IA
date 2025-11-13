import os, json, pandas as pd, matplotlib.pyplot as plt
import warnings, numpy as np
from plotting import create_feature_plots
import sys

# Configuration parameters
MIN_CORRELATION = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
MIN_CORRELATION = max(0.0, min(1.0, MIN_CORRELATION))
try:
    MOVING_AVG_WINDOW = int(sys.argv[2]) if len(sys.argv) > 2 else 20
except Exception:
    MOVING_AVG_WINDOW = 20
MOVING_AVG_WINDOW = max(3, min(400, MOVING_AVG_WINDOW))
print(f"Moving average window (requested): {MOVING_AVG_WINDOW}")

warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')

# Load and clean data

dfkep = pd.read_csv('data/training/kepler.csv', comment='#', on_bad_lines='skip')
print(f"File: kepler.csv")
print(f"Number of columns: {dfkep.shape[1]}")
print(f"Headings: {dfkep.columns.tolist()}")
print(f"Shape: {dfkep.shape}")
print()

target_col = 'koi_disposition'

# Convert all columns to numeric except target
for col in dfkep.columns:
    if dfkep[col].dtype == 'object' and col != target_col:
        dfkep[col] = pd.to_numeric(dfkep[col], errors='coerce')

# Remove rows with missing target or too many missing values
dfkep = dfkep.dropna(subset=[target_col])
threshold = dfkep.shape[1] * 0.5
dfkep = dfkep.dropna(thresh=threshold)
dfkep.reset_index(drop=True, inplace=True)

print(f"After cleaning kepler.csv: {dfkep.shape[0]} rows, {dfkep.shape[1]} columns")
numeric_cols = sum(1 for col in dfkep.columns if dfkep[col].dtype in ['int64', 'float64'])
print(f"  Numeric columns: {numeric_cols}")
print()

# Encode target variable
Ykep = dfkep[target_col]
ordered_targets = ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']
ordered_targets = [t for t in ordered_targets if t in Ykep.unique()]

label_to_idx = {label: idx for idx, label in enumerate(ordered_targets)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

encoded_target = Ykep.map(label_to_idx)

print(f"ENCODED_TARGET_KEY_JSON kepler.csv: " + json.dumps({str(k): v for k, v in idx_to_label.items()}))
print()

# Feature correlation analysis
print("Feature-target correlation analysis for kepler.csv:")
feature_corr = {}

for col in dfkep.columns:
    if col == target_col:
        continue
    if dfkep[col].dtype in ['int64', 'float64']:
        try:
            unique_vals = dfkep[col].dropna().unique()
            # Skip binary features (need at least 3 unique values for meaningful correlation)
            if len(unique_vals) < 3:
                print(f"Skipping {col}: only {len(unique_vals)} unique values (minimum 3 required)")
                continue
            
            corr = dfkep[col].corr(encoded_target)
            if not pd.isna(corr):
                feature_corr[col] = abs(corr)
        except Exception as e:
            print(f"Error calculating correlation for {col}: {e}")

if not feature_corr:
    print(f"No valid correlations found for kepler.csv")
    important_features = []
else:
    # Sort features by correlation strength
    sorted_features = sorted(feature_corr.items(), key=lambda x: x[1], reverse=True)
    
    # Filter features by correlation threshold
    thresholded = [(feat, corr) for feat, corr in sorted_features if corr >= MIN_CORRELATION]
    
    print(f"Top {len(thresholded)} features (>= {MIN_CORRELATION:.2f}) for kepler.csv:" if thresholded else f"Top 0 features (no features >= {MIN_CORRELATION:.2f}) for kepler.csv:")
    for idx, (feat_name, corr_val) in enumerate(thresholded, start=1):
        print(f"  {idx}. {feat_name}: {corr_val:.4f}")
    
    important_features = [f for f, _ in thresholded]
    
    # Generate plots for features meeting the threshold
    if thresholded:
        create_feature_plots(
            dfkep,
            encoded_target,
            thresholded,
            min_data_points=None,
            window_size=MOVING_AVG_WINDOW
        )

print()

# Output feature metadata
print(f"MODEL_FEATURES kepler.csv: " + json.dumps(important_features))

if important_features:
    stats = {}
    for f in important_features:
        series = dfkep[f]
        if series.dtype in ['int64', 'float64'] and len(series.dropna()) > 0:
            stats[f] = {
                "min": float(series.min()),
                "max": float(series.max())
            }
    print(f"STATS_FEATURES kepler.csv: " + json.dumps(stats))

# Generate curves for classification
if important_features:
    try:
        curves = {}
        
        for f in important_features:
            series_x = dfkep[f]
            mask = series_x.notna() & encoded_target.notna()
            x_vals = series_x[mask].to_numpy()
            y_vals = encoded_target[mask].to_numpy()
            
            if len(x_vals) < 3:
                continue
            
            # Sort by feature values
            order = np.argsort(x_vals)
            x_sorted = x_vals[order]
            y_sorted = y_vals[order]
            
            # Calculate moving average
            window_size = max(3, min(MOVING_AVG_WINDOW, len(y_sorted)))
            if window_size > len(y_sorted):
                window_size = len(y_sorted)
            if window_size < 3 or len(y_sorted) < window_size:
                continue
            ma = np.convolve(y_sorted, np.ones(window_size) / window_size, mode='valid')
            x_ma = x_sorted[window_size - 1:]
            
            # Downsample if too many points
            if len(x_ma) > 400:
                step = len(x_ma) // 400
                x_ma = x_ma[::step]
                ma = ma[::step]
            
            curves[f] = {
                "x": x_ma.tolist(),
                "y": ma.tolist(),
                "window": window_size
            }
        
        # Save curves to JSON file
        curves_payload = {
            "classes": ordered_targets,
            "idx_to_label": idx_to_label,
            "label_to_idx": label_to_idx,
            "features": curves,
            "min_correlation": MIN_CORRELATION,
            "moving_avg_window": MOVING_AVG_WINDOW
        }
        
        curves_path = os.path.join(os.path.dirname(__file__), 'combined_curves.json')
        with open(curves_path, 'w', encoding='utf-8') as f:
            json.dump(curves_payload, f, ensure_ascii=False, indent=2)
        
        print(f"CURVES_FILE kepler.csv: combined_curves.json")
        
    except Exception as e:
        print(f"Error generating curves: {e}")