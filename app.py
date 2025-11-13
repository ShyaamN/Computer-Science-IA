import os, re, sys, subprocess, json, bisect
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
app = Flask(__name__)
LAST_RUN = None
TRAINED_MODELS = {}
COMBINED_CURVES = None

MIN_CORR_BOUNDS = (0.0, 1.0)
MOVING_WINDOW_BOUNDS = (3, 400)

#------------------------------------------------------------------------------

def get_settings():
    config_path = os.path.join(os.path.dirname(__file__), 'dashboard_config.json')
    defaults = {"min_correlation": 0.3, "moving_avg_window": 20}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                mc = float(cfg.get("min_correlation", defaults["min_correlation"]))
                window = int(cfg.get("moving_avg_window", defaults["moving_avg_window"]))
                mc = max(MIN_CORR_BOUNDS[0], min(MIN_CORR_BOUNDS[1], mc))
                window = max(MOVING_WINDOW_BOUNDS[0], min(MOVING_WINDOW_BOUNDS[1], window))
                return {"min_correlation": mc, "moving_avg_window": window}
        except Exception:
            pass
    return defaults.copy()

def run_main_script():
    settings = get_settings()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable,
        "main.py",
        str(settings["min_correlation"]),
        str(settings["moving_avg_window"])
    ]
    started = datetime.now()
    result = subprocess.run(cmd, cwd=base_dir, capture_output=True, text=True, shell=False)
    finished = datetime.now()
    duration_ms = int((finished - started).total_seconds() * 1000)
    return {"stdout": result.stdout.strip(), "stderr": result.stderr.strip(), "returncode": result.returncode, "started": started, "finished": finished, "duration_ms": duration_ms}

#------------------------------------------------------------------------------

def parse_output(text):
    datasets, order, cur_ds = {}, [], None
    def normalize(name):
        return name.strip().rstrip(":").strip() if name else name
    def ensure(name):
        nonlocal datasets, order
        name = normalize(name)
        if name not in datasets:
            datasets[name] = {"num_columns": None, "shape": None, "headings": [], "rows": None, "cols": None, "numeric_columns": None, "dropped_features": 0, "top_features": [], "plot_path": None, "model_features": [], "feature_stats": {}, "target_key": {}, "feature_plots": {}}
            order.append(name)
        return datasets[name]
    
    for raw in text.splitlines():
        line = raw.strip()
        if len(line) == 0:
            continue
        
        if line.startswith("File: "):
            file_part = normalize(line[6:])
            # Skip the "(curves)" variant - only use the main dataset
            if "(curves)" not in file_part:
                cur_ds = file_part
                ensure(cur_ds)
        elif line.startswith("Number of columns: ") and cur_ds:
            try:
                ensure(cur_ds)["num_columns"] = int(line.split(":", 1)[1].strip())
            except: pass
        elif line.startswith("Headings: ") and cur_ds:
            content = line.split(":", 1)[1].strip()
            items = []
            if "[" in content and "]" in content:
                inner = content[content.index("[")+1:content.rindex("]")]
                for p in inner.split(","):
                    cleaned = p.strip().strip(chr(39)).strip(chr(34))
                    if len(cleaned) > 0:
                        items.append(cleaned)
            ensure(cur_ds)["headings"] = items
        elif line.startswith("Shape: ") and cur_ds:
            m = re.search(r"\((\d+)\s*,\s*(\d+)\)", line)
            if m != None:
                r, c = int(m.group(1)), int(m.group(2))
                info = ensure(cur_ds)
                info["shape"], info["rows"], info["cols"] = (r, c), r, c
        elif line.startswith("Numeric columns: ") and cur_ds:
            try:
                ensure(cur_ds)["numeric_columns"] = int(line.split(":", 1)[1].strip())
            except: pass
        elif line.startswith("PLOT: "):
            try:
                _, rest = line.split(":", 1)
                left, right = rest.split("->", 1)
                ds_name = normalize(left.strip())
                path_url = right.strip().replace("\\", "/")
                ensure(ds_name)["plot_path"] = path_url
            except: pass
        elif line.startswith("FEATURE_PLOT: "):
            # Example: "FEATURE_PLOT: kepler.csv -> koi_score -> static/plots/kepler_feature_koi_score.png"
            try:
                _, rest = line.split(":", 1)
                parts = rest.split("->")
                if len(parts) >= 3:
                    ds_name = normalize(parts[0].strip())
                    feature_name = parts[1].strip()
                    path_url = parts[2].strip().replace("\\", "/")
                    ensure(ds_name)["feature_plots"][feature_name] = path_url
            except: pass
        elif line.startswith("MODEL_FEATURES "):
            m = re.match(r"MODEL_FEATURES\s+([^:]+):\s*(\[.*\])", line)
            if m != None:
                ds = normalize(m.group(1))
                try:
                    feats = json.loads(m.group(2))
                    ensure(ds)["model_features"] = feats
                except: pass
        elif line.startswith("STATS_FEATURES "):
            m = re.match(r"STATS_FEATURES\s+([^:]+):\s*(\{.*\})", line)
            if m != None:
                ds = normalize(m.group(1))
                try:
                    stats_obj = json.loads(m.group(2))
                    ensure(ds)["feature_stats"] = stats_obj
                except: pass
        elif line.startswith("ENCODED_TARGET_KEY_JSON "):
            m = re.match(r"ENCODED_TARGET_KEY_JSON\s+([^:]+):\s*(\{.*\})", line)
            if m != None:
                ds = normalize(m.group(1))
                try:
                    parsed = json.loads(m.group(2))
                    converted = {int(k) if k.isdigit() else k: v for k, v in parsed.items()}
                    ensure(ds)["target_key"] = converted
                except: pass
        elif line.startswith("Top ") and " features (" in line and cur_ds:
            # Example: "Top 6 features (>= 0.30) for kepler.csv:"
            m = re.match(r"Top\s+(\d+)\s+features.*", line)
            if m != None:
                ensure(cur_ds)["top_features"] = []
        elif cur_ds and re.match(r"^\s*\d+\.\s+", line):
            # Example: "  1. koi_score: 0.8859"
            m = re.match(r"^\s*(\d+)\.\s+(\S+):\s+([\d.]+)", line)
            if m != None:
                idx, feat, val = m.group(1), m.group(2), float(m.group(3))
                ensure(cur_ds)["top_features"].append((int(idx), feat, val))
    
    return {"order": order, "datasets": datasets}

#------------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def dashboard():
    settings = get_settings()
    min_correlation = settings["min_correlation"]
    moving_avg_window = settings["moving_avg_window"]
    
    # If POST, update settings
    if request.method == 'POST':
        try:
            min_correlation = float(request.form.get('min_correlation', min_correlation))
        except Exception:
            min_correlation = settings["min_correlation"]
        try:
            moving_avg_window = int(request.form.get('moving_avg_window', moving_avg_window))
        except Exception:
            moving_avg_window = settings["moving_avg_window"]

        min_correlation = max(MIN_CORR_BOUNDS[0], min(MIN_CORR_BOUNDS[1], min_correlation))
        moving_avg_window = max(MOVING_WINDOW_BOUNDS[0], min(MOVING_WINDOW_BOUNDS[1], moving_avg_window))

        with open(os.path.join(os.path.dirname(__file__), 'dashboard_config.json'), 'w') as f:
            json.dump({
                "min_correlation": min_correlation,
                "moving_avg_window": moving_avg_window
            }, f)
    
    if LAST_RUN is None:
        return render_template(
            "dashboard.html",
            has_run=False,
            min_correlation=min_correlation,
            moving_avg_window=moving_avg_window
        )
    
    run = LAST_RUN
    output = run["stdout"] or "No output produced"
    error = run["stderr"] if run["returncode"] != 0 else ""
    parsed = parse_output(output)
    return render_template(
        "dashboard.html",
        has_run=True,
        output=output,
        error=error,
        started=run["started"],
        finished=run["finished"],
        duration_ms=run["duration_ms"],
        parsed=parsed,
        min_correlation=min_correlation,
        moving_avg_window=moving_avg_window
    )

def _load_combined_curves():
    global COMBINED_CURVES
    if COMBINED_CURVES != None:
        return COMBINED_CURVES
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combined_curves.json')
    if os.path.exists(path) == True:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                COMBINED_CURVES = json.load(f)
        except Exception:
            COMBINED_CURVES = None
    return COMBINED_CURVES

def _estimate_curve_value(x_val, curve_x, curve_y):
    if len(curve_x) == 0 or len(curve_y) == 0:
        return None
    idx = bisect.bisect_left(curve_x, x_val)
    if idx <= 0:
        return curve_y[0]
    if idx >= len(curve_x):
        return curve_y[-1]
    x0, x1 = curve_x[idx-1], curve_x[idx]
    y0, y1 = curve_y[idx-1], curve_y[idx]
    if x1 == x0:
        return (y0 + y1) / 2.0
    t = (x_val - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

#------------------------------------------------------------------------------

@app.route('/combined_curves', methods=['GET'])
def get_combined_curves():
    curves = _load_combined_curves()
    if len(curves) == 0:
        return jsonify({'error': 'Curve data unavailable'}), 404
    return jsonify(curves)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    curves = _load_combined_curves()
    if len(curves) == 0:
        return jsonify({'error': 'Curve data unavailable'}), 400
    settings = get_settings()
    
    required = curves.get('features', {})
    if len(required) == 0:
        return jsonify({'error': 'No curve features present'}), 400
    
    missing = []
    per_feature = {}
    for feat in required.keys():
        if feat not in data:
            missing.append(feat)
        else:
            try:
                per_feature[feat] = float(data[feat])
            except Exception:
                missing.append(feat)
    
    if len(missing) > 0:
        return jsonify({'error': 'Missing feature values', 'missing': missing}), 400
    
    est_vals = []
    per_feature_estimates = {}
    for feat, curve in required.items():
        x_arr = curve.get('x', [])
        y_arr = curve.get('y', [])
        if len(x_arr) > 0 and len(y_arr) > 0:
            v = _estimate_curve_value(per_feature[feat], x_arr, y_arr)
            if v != None:
                est_vals.append(v)
                per_feature_estimates[feat] = v
    
    if len(est_vals) == 0:
        return jsonify({'error': 'Unable to compute curve estimates'}), 400
    
    avg_encoded = float(np.mean(est_vals))
    idx_to_label = curves.get('idx_to_label', {})
    numeric_indices = sorted([int(k) for k in idx_to_label.keys() if str(k).isdigit()])

    
    if len(numeric_indices) == 0:
        return jsonify({'error': 'Invalid curve class mapping'}), 500
    
    nearest_idx = min(numeric_indices, key=lambda i: abs(i - avg_encoded))
    label = idx_to_label.get(str(nearest_idx), idx_to_label.get(nearest_idx, str(nearest_idx)))
    
    # Generate annotated plots
    annotated_plots = {}
    try:
        import pandas as pd
        from plotting import create_annotated_feature_plot
        
        # Load the dataset
        dfkep = pd.read_csv('data/training/kepler.csv', comment='#', on_bad_lines='skip')
        target_col = 'koi_disposition'
        
        # Convert non-numeric columns
        for col in dfkep.columns:
            if dfkep[col].dtype == 'object' and col != target_col:
                dfkep[col] = pd.to_numeric(dfkep[col], errors='coerce')
        
        # Clean data
        dfkep = dfkep.dropna(subset=[target_col])
        threshold = dfkep.shape[1] * 0.5
        dfkep = dfkep.dropna(thresh=threshold)
        dfkep.reset_index(drop=True, inplace=True)
        
        # Encode target
        Ykep = dfkep[target_col]
        ordered_targets = ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']
        ordered_targets = [t for t in ordered_targets if t in Ykep.unique()]
        label_to_idx_map = {v: i for i, v in enumerate(ordered_targets)}
        encoded_target = Ykep.map(label_to_idx_map)
        
        # Generate annotated plot for each feature
        for feat, user_val in per_feature.items():
            if feat in per_feature_estimates and feat in dfkep.columns:
                plot_path = create_annotated_feature_plot(
                    dfkep,
                    encoded_target,
                    feat,
                    user_val,
                    per_feature_estimates[feat],
                    window_size=settings["moving_avg_window"]
                )
                if plot_path:
                    annotated_plots[feat] = plot_path.replace('\\', '/')
    except Exception as e:
        print(f"Error generating annotated plots: {e}")
    
    return jsonify({
        'prediction': label,
        'avg_encoded': round(avg_encoded, 4),
        'nearest_index': nearest_idx,
        'available_classes': [idx_to_label.get(str(i), idx_to_label.get(i, str(i))) for i in numeric_indices],
        'per_feature': {k: round(float(v), 4) for k, v in per_feature.items()},
        'per_feature_estimates': {k: round(float(v), 4) for k, v in per_feature_estimates.items()},
        'annotated_plots': annotated_plots
    })

#------------------------------------------------------------------------------

@app.route("/run", methods=["POST"])
def run_analysis():
    global LAST_RUN, COMBINED_CURVES
    LAST_RUN = run_main_script()
    COMBINED_CURVES = None  # Force reload of curves on next request
    return redirect(url_for("dashboard"))

@app.route("/update_settings", methods=["POST"])
def update_settings():
    min_corr = request.form.get("min_correlation", "0.3").strip()
    window_raw = request.form.get("moving_avg_window", "20").strip()
    try:
        mc_val = float(min_corr)
        if mc_val < 0 or mc_val > 1:
            raise ValueError
    except Exception:
        flash("Invalid min correlation; must be between 0 and 1.", "error")
        mc_val = 0.3
    try:
        window_val = int(window_raw)
    except Exception:
        window_val = 20
    
    mc_val = max(MIN_CORR_BOUNDS[0], min(MIN_CORR_BOUNDS[1], mc_val))
    window_val = max(MOVING_WINDOW_BOUNDS[0], min(MOVING_WINDOW_BOUNDS[1], window_val))
    config_path = os.path.join(os.path.dirname(__file__), 'dashboard_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            "min_correlation": mc_val,
            "moving_avg_window": window_val
        }, f)
    
    wants_json = request.headers.get("X-Requested-With") == "XMLHttpRequest" or "application/json" in (request.headers.get("Accept") or "")
    if wants_json == True:
        return jsonify({"status": "ok", "min_correlation": mc_val, "moving_avg_window": window_val})
    flash("Settings saved.", "ok")
    return redirect(url_for("dashboard"))

#------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
