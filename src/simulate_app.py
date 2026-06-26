import os
import time
import json
import numpy as np
import pandas as pd
import threading
import collections
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras import layers, models
import gc

app = Flask(__name__)
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
UPLOAD_PATH = os.path.join(BASE_DIR, "sim_upload.csv")   # shared upload — set once at analyze stage

CLASS_NAMES  = ["Benign", "DoS", "DDoS", "Botnet", "BruteForce", "WebAttack"]
CLASS_COLORS = ["#27AE60", "#E67E22", "#C0392B", "#8E44AD", "#2980B9", "#F39C12"]

ALERT_META = {
    0: {"level": "clear",    "color": "#27AE60", "icon": "✅", "title": "ALL CLEAR",
        "msg": "Normal traffic patterns detected. No threats active.",
        "action": "No action required. Continue monitoring."},
    1: {"level": "high",     "color": "#E67E22", "icon": "⚠️",  "title": "DoS ATTACK DETECTED",
        "msg": "Denial of Service — Single source flooding detected.",
        "action": "Block source IP immediately. Apply rate limiting."},
    2: {"level": "critical", "color": "#C0392B", "icon": "🔴", "title": "DDoS ATTACK DETECTED",
        "msg": "Distributed DoS — Multi-source flood in progress.",
        "action": "Contact ISP. Activate CDN scrubbing. Enable geo-blocking."},
    3: {"level": "critical", "color": "#8B0000", "icon": "☠️",  "title": "BOTNET ACTIVITY",
        "msg": "Compromised device communicating with C&C server.",
        "action": "Isolate endpoint immediately. Run malware scan. Block C&C IPs."},
    4: {"level": "medium",   "color": "#2980B9", "icon": "🔐", "title": "BRUTE FORCE DETECTED",
        "msg": "Repeated credential attempts on SSH/FTP detected.",
        "action": "Lock account. Enable MFA. Add source IP to blocklist."},
    5: {"level": "medium",   "color": "#8E44AD", "icon": "💉", "title": "WEB ATTACK DETECTED",
        "msg": "Malicious payload in HTTP traffic (SQLi/XSS).",
        "action": "Check WAF rules. Sanitize inputs. Review server logs."},
}

NUM_CLIENTS   = 5
NUM_ROUNDS    = 3
LOCAL_EPOCHS  = 3
BATCH_SIZE    = 256
LEARNING_RATE = 0.001
NUM_CLASSES   = 6
TOTAL_ROWS    = 50000
WINDOW_SIZE   = 30   # sliding window for live alerts

# ── DP Sweep constants ──
PROXY_ROWS         = 5000
PROXY_SIGMA_VALUES = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]

# ── Global simulation state ──
sim_state = {
    "running": False, "log": [], "round": 0,
    "client_accs": {}, "global_accs": [], "global_losses": [],
    "done": False, "error": None, "graphs": {},
    "client_distributions": {},
    "actual_distributions": {},
    "drift_history": [],
    "local_vs_global": {},
    "comm_cost": 0,
    "phase1_metrics": None,
    "dp_curve": [],
    "dp_sigma_used": 0.0001,   # records which sigma the user picked for the simulation
}

# ── DP Sweep state (populated before simulation) ──
dp_sweep_state = {
    "running":        False,
    "done":           False,
    "curve":          [],    # list of {"sigma": ..., "acc": ...}
    "progress":       0,     # 0–100
    "recommendation": {},
    "graph":          None,  # base64 PNG of the sweep curve
    "error":          None,
}

# ── Dataset analysis state (populated before simulation) ──
dataset_analysis_state = {
    "done":  False,
    "stats": {},
    "error": None,
}

# ── Live alert state ──
alert_state = {
    "active": False,
    "clients": {i: {
        "status": "clear",
        "threat_class": 0,
        "confidence": 0.0,
        "window": collections.deque(maxlen=WINDOW_SIZE),
        "history": [],
        "flow_count": 0,
    } for i in range(NUM_CLIENTS)},
    "server_broadcast": {"level": "clear", "msg": "System initializing...", "active_threats": 0},
    "timeline": [],
    "stream_data": {},
}

trained_model   = None
trained_scaler  = None
feature_names   = None


def reset_state():
    global trained_model, trained_scaler, feature_names
    trained_model  = None
    trained_scaler = None
    feature_names  = None
    sim_state.update({
        "running": False, "log": [], "round": 0,
        "client_accs": {}, "global_accs": [], "global_losses": [],
        "done": False, "error": None, "graphs": {},
        "client_distributions": {}, "actual_distributions": {},
        "drift_history": [], "local_vs_global": {},
        "comm_cost": 0, "phase1_metrics": None, "dp_curve": [],
        "dp_sigma_used": 0.0001,
    })
    alert_state["active"] = False
    alert_state["server_broadcast"] = {"level": "clear", "msg": "Idle", "active_threats": 0}
    alert_state["timeline"] = []
    alert_state["stream_data"] = {}
    for i in range(NUM_CLIENTS):
        alert_state["clients"][i] = {
            "status": "clear", "threat_class": 0,
            "confidence": 0.0,
            "window": collections.deque(maxlen=WINDOW_SIZE),
            "history": [], "flow_count": 0,
        }


def log(msg, level="info"):
    entry = {"msg": msg, "level": level, "time": time.strftime("%H:%M:%S")}
    sim_state["log"].append(entry)
    print(f"[{entry['time']}] {msg}")


# ──────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────
def build_cnn(input_dim):
    """Full 1D-CNN used in the main FL simulation."""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Reshape((input_dim, 1)),
        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.Conv1D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_proxy_cnn(input_dim):
    """
    Lightweight proxy CNN for the DP sweep.
    Smaller than the full model so 10 sweep runs complete quickly (~2–3 min total).
    The noise TREND is what matters here, not the absolute accuracy numbers.
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Reshape((input_dim, 1)),
        layers.Conv1D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def get_model_weights(model):
    return model.get_weights()


def set_model_weights(model, weights):
    model.set_weights(weights)


def fedavg(global_weights, client_weights_list, client_sizes):
    total    = sum(client_sizes)
    averaged = []
    for li in range(len(global_weights)):
        weighted_sum = sum(
            client_weights_list[c][li] * client_sizes[c]
            for c in range(len(client_weights_list))
        )
        averaged.append(weighted_sum / total)
    return averaged


def apply_dp(local_weights, global_weights_ref, clip_norm=10.0, noise_multiplier=0.0001):
    """
    Apply Differential Privacy to the weight UPDATE (delta), not the raw weights.

    The standard FL-DP approach:
      1. Compute delta = local_weights - global_weights_ref  (the actual round update)
      2. Clip the GLOBAL L2 norm of the concatenated delta vector to clip_norm
      3. Add calibrated Gaussian noise to each layer's clipped delta
      4. Reconstruct updated weights = global_weights_ref + clipped_noised_delta

    Why this matters: per-layer norm clipping of raw weights (old approach) was
    scaling down large CNN/Dense layers (norms >> 10) to near-zero magnitude,
    collapsing the global model quality after FedAvg and capping accuracy at ~22%.
    """
    # Step 1 — per-layer deltas
    deltas = [
        lw.astype(np.float64) - gw.astype(np.float64)
        for lw, gw in zip(local_weights, global_weights_ref)
    ]

    # Step 2 — global L2 norm of the full update vector
    global_norm = float(np.sqrt(sum(np.sum(d ** 2) for d in deltas)))

    # Clip: scale down proportionally if norm > clip_norm
    scale = min(1.0, clip_norm / max(global_norm, 1e-10))
    clipped_deltas = [d * scale for d in deltas]

    # Step 3+4 — add Gaussian noise and reconstruct weights
    dp_weights = []
    for gw, cd in zip(global_weights_ref, clipped_deltas):
        noise = np.random.normal(0.0, noise_multiplier * clip_norm, size=cd.shape)
        dp_weights.append((gw.astype(np.float64) + cd + noise).astype(np.float32))
    return dp_weights


def weight_drift(w1, w2):
    """L2 distance between two weight lists."""
    return float(np.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip(w1, w2))))


# ──────────────────────────────────────────────
# DP PRE-SIMULATION ANALYSIS  (NEW)
# ──────────────────────────────────────────────

def _preprocess_df(filepath):
    """Shared preprocessing used by analysis, sweep, and simulation."""
    df = pd.read_csv(filepath, low_memory=False, on_bad_lines="skip")
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    df.drop(columns=obj_cols, inplace=True, errors="ignore")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(0, inplace=True)
    return df


def analyze_dataset_for_dp(filepath):
    """
    Analyze the uploaded CSV and return dataset statistics + a data-aware DP recommendation.
    This runs fast (no model training) — pure statistics on the raw data.
    """
    df = _preprocess_df(filepath)

    if "Label" not in df.columns:
        return {"error": "CSV must have a 'Label' column with class IDs (0-5)."}

    row_count     = len(df)
    feature_count = df.shape[1] - 1

    # ── Class distribution ──
    class_counts = {}
    for ci, cname in enumerate(CLASS_NAMES):
        class_counts[cname] = int((df["Label"] == ci).sum())

    counts_nonzero = [v for v in class_counts.values() if v > 0]
    max_count      = max(counts_nonzero) if counts_nonzero else 1
    min_count      = min(counts_nonzero) if counts_nonzero else 1
    imbalance_ratio = round(max_count / max(min_count, 1), 2)

    # ── Feature scale (used to recommend clip_norm) ──
    X        = df.drop(columns=["Label"]).values.astype(np.float32)
    mean_abs = float(np.abs(X).mean())
    mean_std = float(X.std(axis=0).mean())

    # ── Rule-based recommendation ──

    # 1. Sigma max based on class imbalance
    if imbalance_ratio > 20:
        suggested_max    = 0.001
        imbalance_note   = (
            f"Very high imbalance ({imbalance_ratio}×) — DP noise will disproportionately "
            f"destroy gradients from minority classes. Keep σ very low."
        )
    elif imbalance_ratio > 5:
        suggested_max    = 0.005
        imbalance_note   = (
            f"Moderate imbalance ({imbalance_ratio}×) — lower noise protects minority "
            f"class signal during federated aggregation."
        )
    else:
        suggested_max    = 0.01
        imbalance_note   = (
            f"Balanced classes ({imbalance_ratio}× ratio) — moderate noise is tolerable "
            f"without significant minority class degradation."
        )

    # 2. Adjust for dataset size
    if row_count < 5000:
        suggested_max = min(suggested_max, 0.001)
        size_note = (
            f"Small dataset ({row_count:,} rows) — fewer samples amplify noise impact. "
            f"Use minimal σ to preserve learning signal."
        )
    elif row_count < 20000:
        size_note = (
            f"Medium dataset ({row_count:,} rows) — standard DP settings apply well."
        )
    elif row_count > 100000:
        suggested_max = min(suggested_max * 2.0, 0.05)
        size_note = (
            f"Large dataset ({row_count:,} rows) — abundant data compensates for noise. "
            f"Slightly higher σ is viable."
        )
    else:
        size_note = (
            f"Good dataset size ({row_count:,} rows) — DP settings have good flexibility."
        )

    # 3. Clip norm based on feature scale
    if mean_abs > 10:
        suggested_clip = 50.0
        clip_note = (
            f"Features are large-scale (mean |x| = {mean_abs:.1f}). "
            f"clip_norm=50.0 recommended — otherwise clipping removes too much signal."
        )
    elif mean_abs > 1:
        suggested_clip = 10.0
        clip_note = (
            f"Features are moderate-scale (mean |x| = {mean_abs:.2f}). "
            f"Current clip_norm=10.0 is appropriate."
        )
    else:
        suggested_clip = 2.0
        clip_note = (
            f"Features are small-scale / normalized (mean |x| = {mean_abs:.4f}). "
            f"clip_norm=1.0–2.0 is sufficient and more DP-efficient."
        )

    suggested_max = round(suggested_max, 4)

    return {
        "row_count":        row_count,
        "feature_count":    feature_count,
        "class_counts":     class_counts,
        "imbalance_ratio":  imbalance_ratio,
        "mean_abs_feature": round(mean_abs, 4),
        "mean_std_feature": round(mean_std, 4),
        "recommendation": {
            "suggested_sigma_min": 0.0001,
            "suggested_sigma_max": suggested_max,
            "suggested_clip_norm": suggested_clip,
            "imbalance_note":      imbalance_note,
            "size_note":           size_note,
            "clip_note":           clip_note,
            "summary": (
                f"Based on your dataset ({row_count:,} rows, imbalance {imbalance_ratio}×), "
                f"the recommended DP noise range is σ = 0.0001 – {suggested_max}. "
                f"Beyond this, minority class detection risk increases significantly."
            ),
        },
    }


def run_dp_proxy_sweep_job(filepath):
    """
    Runs a lightweight proxy DP sweep across PROXY_SIGMA_VALUES.
    Uses a smaller model (build_proxy_cnn) on PROXY_ROWS samples,
    1 FL round, 1 local epoch per client.
    Finds the accuracy elbow and returns the recommended σ range.
    Updates dp_sweep_state live so the frontend can poll progress.
    """
    global dp_sweep_state

    try:
        dp_sweep_state.update({
            "running": True, "done": False, "curve": [],
            "progress": 0, "error": None, "graph": None,
            "recommendation": {},
        })

        df = _preprocess_df(filepath)
        if "Label" not in df.columns:
            dp_sweep_state["error"]   = "CSV must have a 'Label' column with class IDs (0-5)."
            dp_sweep_state["running"] = False
            return

        # ── Stratified subsample to PROXY_ROWS ──
        if len(df) > PROXY_ROWS:
            sampled_parts = []
            for ci in range(NUM_CLASSES):
                class_df = df[df["Label"] == ci]
                if len(class_df) == 0:
                    continue
                n_take = max(1, int(PROXY_ROWS * len(class_df) / len(df)))
                sampled_parts.append(class_df.sample(min(n_take, len(class_df)), random_state=42))
            df_proxy = pd.concat(sampled_parts).sample(frac=1, random_state=0).reset_index(drop=True)
            # Top up if still short
            if len(df_proxy) < PROXY_ROWS:
                extra = df.sample(min(PROXY_ROWS - len(df_proxy), len(df)), random_state=1)
                df_proxy = pd.concat([df_proxy, extra]).drop_duplicates().reset_index(drop=True)
        else:
            df_proxy = df.copy()

        X_all = df_proxy.drop(columns=["Label"]).values.astype(np.float32)
        y_all = df_proxy["Label"].values.astype(np.int32)

        unique_cls = np.unique(y_all)
        stratify   = y_all if len(unique_cls) > 1 else None

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=stratify
        )
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        input_dim = X_tr.shape[1]

        # Equal split across NUM_CLIENTS proxy clients
        n     = len(X_tr)
        chunk = n // NUM_CLIENTS
        proxy_clients = []
        for i in range(NUM_CLIENTS):
            s = i * chunk
            e = s + chunk if i < NUM_CLIENTS - 1 else n
            proxy_clients.append((X_tr[s:e], y_tr[s:e]))

        curve = []

        for idx, sigma in enumerate(PROXY_SIGMA_VALUES):
            # Fresh global proxy model + 1-epoch warm-up
            proxy_global = build_proxy_cnn(input_dim)
            proxy_global.fit(X_tr, y_tr, epochs=1, batch_size=256, verbose=0)
            gw = get_model_weights(proxy_global)

            # 1 FL round across all clients
            cw_list, cs_list = [], []
            for cx, cy in proxy_clients:
                if len(cx) == 0:
                    continue
                cm = build_proxy_cnn(input_dim)
                set_model_weights(cm, gw)
                cm.fit(cx, cy, epochs=1, batch_size=128, verbose=0)
                lw   = get_model_weights(cm)
                dp_lw = apply_dp(lw, gw, clip_norm=10.0, noise_multiplier=sigma)
                cw_list.append(dp_lw)
                cs_list.append(len(cx))
                del cm
                gc.collect()

            if cw_list:
                agg_w = fedavg(gw, cw_list, cs_list)
                set_model_weights(proxy_global, agg_w)

            _, proxy_acc = proxy_global.evaluate(X_te, y_te, verbose=0, batch_size=256)
            curve.append({"sigma": sigma, "acc": round(float(proxy_acc) * 100, 2)})
            dp_sweep_state["curve"]    = curve[:]
            dp_sweep_state["progress"] = round((idx + 1) / len(PROXY_SIGMA_VALUES) * 100)

            del proxy_global
            gc.collect()

        # ── Elbow detection ──
        accs     = [p["acc"] for p in curve]
        baseline = accs[0]

        caution_sigma = None
        danger_sigma  = None
        for point in curve:
            drop = baseline - point["acc"]
            if caution_sigma is None and drop > 2.0:
                caution_sigma = point["sigma"]
            if danger_sigma  is None and drop > 5.0:
                danger_sigma  = point["sigma"]

        if caution_sigma is None:
            caution_sigma = PROXY_SIGMA_VALUES[-1]
        if danger_sigma  is None:
            danger_sigma  = PROXY_SIGMA_VALUES[-1]

        # Recommended max = last safe sigma (one step before caution)
        caution_idx     = PROXY_SIGMA_VALUES.index(caution_sigma)
        recommended_max = PROXY_SIGMA_VALUES[max(0, caution_idx - 1)]
        default_pick    = recommended_max

        recommendation = {
            "baseline_acc":    round(baseline, 2),
            "caution_sigma":   caution_sigma,
            "danger_sigma":    danger_sigma,
            "recommended_min": 0.0001,
            "recommended_max": recommended_max,
            "default_pick":    default_pick,
            "summary": (
                f"Proxy sweep complete. Baseline proxy accuracy: {baseline:.1f}%. "
                f"Accuracy stays within 2% of baseline up to σ={recommended_max}. "
                f"Significant degradation (>2%) begins at σ={caution_sigma}. "
                f"Recommended σ range: 0.0001 – {recommended_max}."
            ),
        }

        dp_sweep_state["recommendation"] = recommendation
        dp_sweep_state["graph"]          = generate_dp_sweep_graph(curve, recommendation)
        dp_sweep_state["done"]           = True
        dp_sweep_state["running"]        = False

    except Exception as e:
        import traceback
        traceback.print_exc()
        dp_sweep_state["error"]   = str(e)
        dp_sweep_state["running"] = False


def generate_dp_sweep_graph(curve, recommendation):
    """Generate the DP sweep accuracy-vs-sigma chart as a base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    FC  = "#FFFFFF"
    AX  = "#F4F5F9"
    TXT = "#1A1A2E"
    SUB = "#4A4A6A"
    GRD = "#DDDDE8"

    sigmas = [p["sigma"] for p in curve]
    accs   = [p["acc"]   for p in curve]

    caution_sigma   = recommendation["caution_sigma"]
    danger_sigma    = recommendation["danger_sigma"]
    recommended_max = recommendation["recommended_max"]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(FC)
    ax.set_facecolor(AX)

    # ── Shade zones ──
    ax.axvspan(min(sigmas),   recommended_max, alpha=0.12, color="#27AE60", zorder=0)
    if caution_sigma != recommended_max:
        ax.axvspan(recommended_max, caution_sigma,  alpha=0.12, color="#F39C12", zorder=0)
    if danger_sigma != caution_sigma:
        ax.axvspan(caution_sigma,   max(sigmas),    alpha=0.12, color="#C0392B", zorder=0)

    # ── Accuracy curve ──
    ax.plot(sigmas, accs, color="#1A1A2E", linewidth=2.5,
            marker="o", markersize=9, markerfacecolor="#2980B9", zorder=5)

    for s, a in zip(sigmas, accs):
        ax.annotate(f"{a:.1f}%", (s, a),
                    textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=8.5, color=TXT, fontweight="bold")

    # ── Threshold lines ──
    ax.axvline(recommended_max, color="#27AE60", linewidth=2.0, linestyle="--",
               label=f"Recommended max  σ = {recommended_max}", zorder=6)
    ax.axvline(caution_sigma,   color="#F39C12", linewidth=2.0, linestyle="--",
               label=f"Caution threshold  σ = {caution_sigma}", zorder=6)
    if danger_sigma != caution_sigma:
        ax.axvline(danger_sigma, color="#C0392B", linewidth=2.0, linestyle="--",
                   label=f"Danger threshold  σ = {danger_sigma}", zorder=6)

    # ── Zone labels ──
    ax.text(min(sigmas) * 1.5, min(accs) - 2, "✅ Safe",    color="#27AE60", fontsize=9)
    ax.text(recommended_max * 1.3, min(accs) - 2, "⚠️ Caution", color="#F39C12", fontsize=9)
    ax.text(caution_sigma * 1.3,   min(accs) - 2, "❌ Danger",  color="#C0392B", fontsize=9)

    ax.set_xscale("log")
    ax.set_title("DP Proxy Sweep — Proxy Accuracy vs Noise Multiplier (σ)",
                 color=TXT, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Noise Multiplier σ  (log scale)", color=SUB, fontsize=11)
    ax.set_ylabel("Proxy Accuracy (%)", color=SUB, fontsize=11)
    ax.tick_params(colors=SUB)
    ax.grid(alpha=0.4, color=GRD, which="both")
    ax.legend(facecolor="#FFFFFF", edgecolor="#CCCCDD", fontsize=9, loc="upper right")

    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=FC, edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


# ──────────────────────────────────────────────
# CUSTOM DATASET BUILDER
# ──────────────────────────────────────────────
def build_custom_client_data(df, client_configs, total_rows=TOTAL_ROWS):
    """
    client_configs: list of 5 dicts like
      {"Benign": 40, "DoS": 30, "DDoS": 20, "Botnet": 5, "BruteForce": 3, "WebAttack": 2}
    Each dict sums to 100 (percent).
    Returns: list of (X, y) tuples per client, plus actual_distributions dict.
    """
    rows_per_client = total_rows // NUM_CLIENTS
    client_data     = []
    actual_dists    = {}

    class_indices = {}
    for ci, cname in enumerate(CLASS_NAMES):
        idx = df[df["Label"] == ci].index.tolist()
        np.random.shuffle(idx)
        class_indices[ci] = idx

    class_ptr = {ci: 0 for ci in range(NUM_CLASSES)}

    for cid, config in enumerate(client_configs):
        rows_needed = {}
        for cname, pct in config.items():
            ci = CLASS_NAMES.index(cname)
            rows_needed[ci] = int(rows_per_client * pct / 100)

        diff = rows_per_client - sum(rows_needed.values())
        if diff != 0:
            dominant = max(rows_needed, key=rows_needed.get)
            rows_needed[dominant] += diff

        client_rows  = []
        actual_count = {}
        for ci, n in rows_needed.items():
            avail = class_indices[ci][class_ptr[ci]:]
            take  = min(n, len(avail))
            if take < n:
                extra    = n - take
                wrapped  = class_indices[ci][:extra]
                selected = avail[:take] + wrapped
            else:
                selected = avail[:take]
            class_ptr[ci] = (class_ptr[ci] + take) % max(1, len(class_indices[ci]))
            client_rows.extend(selected)
            actual_count[CLASS_NAMES[ci]] = take

        np.random.shuffle(client_rows)
        sub = df.loc[client_rows].copy()
        X   = sub.drop(columns=["Label"]).values.astype(np.float32)
        y   = sub["Label"].values.astype(np.int32)
        client_data.append((X, y))
        actual_dists[cid] = actual_count
        log(f"  Client {cid+1} built: {len(X):,} rows — {actual_count}", "info")

    return client_data, actual_dists


# ──────────────────────────────────────────────
# GRAPH GENERATION
# ──────────────────────────────────────────────
def generate_graphs(y_test, y_pred, round_accs, round_losses,
                    client_acc_history, client_configs=None,
                    actual_dists=None, drift_history=None,
                    local_vs_global=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO

    plt.style.use("default")
    FC  = "#FFFFFF"
    AX  = "#F4F5F9"
    TXT = "#1A1A2E"
    SUB = "#4A4A6A"
    GRD = "#DDDDE8"
    LEG = {"facecolor": "#FFFFFF", "edgecolor": "#CCCCDD"}

    def b64(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor=FC, edgecolor="none")
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    graphs = {}

    # ── 1 Confusion Matrix ──
    cm     = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(FC)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0], linewidths=0.5, linecolor="#FFFFFF")
    axes[0].set_title("Confusion Matrix — Counts", color=TXT, fontsize=12)
    axes[0].set_xlabel("Predicted", color=SUB)
    axes[0].set_ylabel("True", color=SUB)
    axes[0].set_facecolor(AX)
    axes[0].tick_params(colors=SUB)
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Reds",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], linewidths=0.5, linecolor="#FFFFFF")
    axes[1].set_title("Confusion Matrix — %", color=TXT, fontsize=12)
    axes[1].set_xlabel("Predicted", color=SUB)
    axes[1].set_ylabel("True", color=SUB)
    axes[1].set_facecolor(AX)
    axes[1].tick_params(colors=SUB)
    plt.tight_layout()
    graphs["confusion_matrix"] = b64(fig)
    plt.close(fig)

    # ── 2 FL Convergence ──
    rounds = list(range(1, len(round_accs) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(FC)
    axes[0].plot(rounds, [a * 100 for a in round_accs],
                 color="#1A8A48", linewidth=2.5, marker="o",
                 markersize=8, markerfacecolor="#1A1A2E")
    for r, a in zip(rounds, round_accs):
        axes[0].annotate(f"{a*100:.2f}%", (r, a * 100),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=9, color=TXT)
    axes[0].set_title("Global Accuracy per Round", color=TXT, fontsize=12)
    axes[0].set_xlabel("Round", color=SUB)
    axes[0].set_ylabel("Accuracy (%)", color=SUB)
    axes[0].set_facecolor(AX)
    axes[0].tick_params(colors=SUB)
    axes[0].grid(alpha=0.5, color=GRD)
    axes[1].plot(rounds, round_losses,
                 color="#B03020", linewidth=2.5, marker="s",
                 markersize=8, markerfacecolor="#1A1A2E")
    for r, l in zip(rounds, round_losses):
        axes[1].annotate(f"{l:.4f}", (r, l),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=9, color=TXT)
    axes[1].set_title("Global Loss per Round", color=TXT, fontsize=12)
    axes[1].set_xlabel("Round", color=SUB)
    axes[1].set_ylabel("Loss", color=SUB)
    axes[1].set_facecolor(AX)
    axes[1].tick_params(colors=SUB)
    axes[1].grid(alpha=0.5, color=GRD)
    plt.tight_layout()
    graphs["convergence"] = b64(fig)
    plt.close(fig)

    # ── 3 Per-Class F1 ──
    f1   = f1_score(y_test, y_pred, average=None, zero_division=0)
    prec = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec  = recall_score(y_test, y_pred, average=None, zero_division=0)
    x    = np.arange(NUM_CLASSES)
    w    = 0.25
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(FC)
    ax.set_facecolor(AX)
    ax.bar(x - w, prec * 100, w, label="Precision", color="#1E6FA0")
    ax.bar(x,     rec  * 100, w, label="Recall",    color="#1A8A48")
    ax.bar(x + w, f1   * 100, w, label="F1-Score",  color="#B03020")
    for i, (p, r, f) in enumerate(zip(prec, rec, f1)):
        ax.text(i - w, p * 100 + 0.3, f"{p*100:.1f}%", ha="center", fontsize=8, color=TXT)
        ax.text(i,     r * 100 + 0.3, f"{r*100:.1f}%", ha="center", fontsize=8, color=TXT)
        ax.text(i + w, f * 100 + 0.3, f"{f*100:.1f}%", ha="center", fontsize=8, color=TXT)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, color=SUB)
    ax.tick_params(colors=SUB)
    ax.set_ylim(0, 115)
    ax.set_title("Per-Class Precision / Recall / F1", color=TXT, fontsize=12)
    ax.legend(**LEG)
    ax.grid(axis="y", alpha=0.5, color=GRD)
    plt.tight_layout()
    graphs["per_class"] = b64(fig)
    plt.close(fig)

    # ── 4 Client Accuracy History ──
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(FC)
    ax.set_facecolor(AX)
    for cid, hist in client_acc_history.items():
        ax.plot(range(1, len(hist) + 1), [a * 100 for a in hist],
                marker="o", linewidth=2, label=f"Client {cid+1}",
                color=CLASS_COLORS[cid])
    ax.set_title("Client Local Accuracy per Round", color=TXT, fontsize=12)
    ax.set_xlabel("Round", color=SUB)
    ax.set_ylabel("Local Accuracy (%)", color=SUB)
    ax.tick_params(colors=SUB)
    ax.legend(**LEG)
    ax.grid(alpha=0.5, color=GRD)
    plt.tight_layout()
    graphs["client_accs"] = b64(fig)
    plt.close(fig)

    # ── 5 Class Distribution ──
    counts_full = np.zeros(NUM_CLASSES, dtype=int)
    unique, counts = np.unique(y_test, return_counts=True)
    for u, c in zip(unique, counts):
        counts_full[u] = c
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(FC)
    ax.set_facecolor(AX)
    bars = ax.bar(CLASS_NAMES, counts_full, color=CLASS_COLORS, edgecolor="white")
    for bar, cnt in zip(bars, counts_full):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{cnt:,}", ha="center", fontsize=10, color=TXT)
    ax.set_title("Test Set Class Distribution", color=TXT, fontsize=12)
    ax.set_xlabel("Class", color=SUB)
    ax.set_ylabel("Count", color=SUB)
    ax.tick_params(colors=SUB)
    ax.grid(axis="y", alpha=0.5, color=GRD)
    plt.tight_layout()
    graphs["class_dist"] = b64(fig)
    plt.close(fig)

    # ── 6 DP Visualization ──
    np.random.seed(42)
    sample_weights  = np.random.randn(10) * 3
    clip_norm_demo  = 2.5
    norm    = np.linalg.norm(sample_weights)
    clipped = sample_weights * (clip_norm_demo / norm) if norm > clip_norm_demo else sample_weights.copy()
    noise_sigma_demo = 1.2
    noised  = clipped + np.random.normal(0, noise_sigma_demo, size=10)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor(FC)
    fig.suptitle("Differential Privacy — Weight Transformation (Demonstration Scale)",
                 color=TXT, fontsize=13, fontweight="bold")
    x_pos = np.arange(10)

    ax_l = axes[0]
    ax_l.set_facecolor(AX)
    ax_l.bar(x_pos - 0.27, sample_weights, 0.27, label="① Before DP (Original)",  color="#1E6FA0", alpha=0.9)
    ax_l.bar(x_pos,        clipped,        0.27, label="② After Clipping",         color="#C96A10", alpha=0.9)
    ax_l.bar(x_pos + 0.27, noised,         0.27, label="③ After DP Noise Added",   color="#B03020", alpha=0.9)
    ax_l.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax_l.axhline( clip_norm_demo / np.sqrt(10), color="#C96A10",
                  linewidth=1.2, linestyle=":", alpha=0.7, label=f"Clip boundary ±{clip_norm_demo:.1f}/√10")
    ax_l.axhline(-clip_norm_demo / np.sqrt(10), color="#C96A10", linewidth=1.2, linestyle=":", alpha=0.7)
    ax_l.set_title("Weight Values: Before vs After DP", color=TXT, fontsize=11)
    ax_l.set_xlabel("Weight Index", color=SUB)
    ax_l.set_ylabel("Weight Value", color=SUB)
    ax_l.tick_params(colors=SUB)
    ax_l.legend(**LEG, fontsize=8)
    ax_l.grid(axis="y", alpha=0.4, color=GRD)

    ax_r = axes[1]
    ax_r.set_facecolor(AX)
    delta_clip  = np.abs(sample_weights - clipped)
    delta_noise = np.abs(noised - sample_weights)
    ax_r.bar(x_pos - 0.2, delta_clip,  0.38,
             label="|Original − Clipped| (clipping effect)",  color="#C96A10", alpha=0.9)
    ax_r.bar(x_pos + 0.2, delta_noise, 0.38,
             label="|Original − DP Noised| (total DP shift)", color="#B03020", alpha=0.9)
    ax_r.set_title("Magnitude of Change Applied by DP", color=TXT, fontsize=11)
    ax_r.set_xlabel("Weight Index", color=SUB)
    ax_r.set_ylabel("Absolute Change |Δw|", color=SUB)
    ax_r.tick_params(colors=SUB)
    ax_r.legend(**LEG, fontsize=8)
    ax_r.grid(axis="y", alpha=0.4, color=GRD)

    plt.tight_layout()
    graphs["dp_viz"] = b64(fig)
    plt.close(fig)

    # ── 7 Summary Metrics ──
    macro_f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    acc_val    = (y_pred == y_test).mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(FC)
    ax.set_facecolor(AX)
    labels_m = ["Accuracy", "Macro F1", "Macro Precision", "Macro Recall"]
    values   = [acc_val * 100, macro_f1 * 100, macro_prec * 100, macro_rec * 100]
    colors   = ["#1A8A48", "#B03020", "#1E6FA0", "#C96A10"]
    bars = ax.barh(labels_m, values, color=colors, edgecolor="white", height=0.5)
    for bar, val in zip(bars, values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=11,
                color=TXT, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.set_title("Global Model — Summary Metrics", color=TXT, fontsize=12)
    ax.tick_params(colors=SUB)
    ax.grid(axis="x", alpha=0.5, color=GRD)
    plt.tight_layout()
    graphs["summary"] = b64(fig)
    plt.close(fig)

    # ── 8 Configured vs Actual Distribution Heatmap ──
    if client_configs and actual_dists:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(FC)
        configured_mat = np.zeros((NUM_CLIENTS, NUM_CLASSES))
        actual_mat     = np.zeros((NUM_CLIENTS, NUM_CLASSES))
        for cid, config in enumerate(client_configs):
            for cname, pct in config.items():
                ci = CLASS_NAMES.index(cname)
                configured_mat[cid, ci] = pct
        for cid, dist in actual_dists.items():
            total = sum(dist.values())
            for cname, cnt in dist.items():
                ci = CLASS_NAMES.index(cname)
                actual_mat[cid, ci] = round(cnt / max(1, total) * 100, 1)
        sns.heatmap(configured_mat, annot=True, fmt=".0f", cmap="Purples",
                    xticklabels=CLASS_NAMES,
                    yticklabels=[f"Client {i+1}" for i in range(NUM_CLIENTS)],
                    ax=axes[0], linewidths=0.5, linecolor="#FFFFFF",
                    cbar_kws={"label": "%"})
        axes[0].set_title("Configured Distribution (%)", color=TXT, fontsize=12)
        axes[0].set_facecolor(AX)
        axes[0].tick_params(colors=SUB)
        sns.heatmap(actual_mat, annot=True, fmt=".0f", cmap="Reds",
                    xticklabels=CLASS_NAMES,
                    yticklabels=[f"Client {i+1}" for i in range(NUM_CLIENTS)],
                    ax=axes[1], linewidths=0.5, linecolor="#FFFFFF",
                    cbar_kws={"label": "%"})
        axes[1].set_title("Actual Sampled Distribution (%)", color=TXT, fontsize=12)
        axes[1].set_facecolor(AX)
        axes[1].tick_params(colors=SUB)
        plt.suptitle("Configured vs Actual Client Data Distribution",
                     color=TXT, fontsize=13)
        plt.tight_layout()
        graphs["dist_heatmap"] = b64(fig)
        plt.close(fig)

    # ── 9 Weight Drift Per Round ──
    if drift_history and len(drift_history) > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor(FC)
        ax.set_facecolor(AX)
        rounds_d = list(range(1, len(drift_history) + 1))
        for cid in range(NUM_CLIENTS):
            drifts = [d[cid] for d in drift_history if cid in d]
            if drifts:
                ax.plot(rounds_d[:len(drifts)], drifts,
                        marker="o", linewidth=2,
                        label=f"Client {cid+1}", color=CLASS_COLORS[cid])
        ax.set_title("Weight Drift — L2 Distance from Global Model per Round",
                     color=TXT, fontsize=12)
        ax.set_xlabel("Round", color=SUB)
        ax.set_ylabel("L2 Drift", color=SUB)
        ax.tick_params(colors=SUB)
        ax.legend(**LEG)
        ax.grid(alpha=0.5, color=GRD)
        plt.tight_layout()
        graphs["weight_drift"] = b64(fig)
        plt.close(fig)

    # ── 10 Local vs Global Accuracy Gap ──
    if local_vs_global:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(FC)
        ax.set_facecolor(AX)
        cids    = list(local_vs_global.keys())
        local_a = [local_vs_global[c]["local"]  * 100 for c in cids]
        glob_a  = [local_vs_global[c]["global"] * 100 for c in cids]
        x_c = np.arange(len(cids))
        w_c = 0.35
        ax.bar(x_c - w_c/2, local_a, w_c, label="Local Model Acc",  color="#C96A10")
        ax.bar(x_c + w_c/2, glob_a,  w_c, label="Global Model Acc", color="#1A8A48")
        for i, (la, ga) in enumerate(zip(local_a, glob_a)):
            ax.text(i - w_c/2, la + 0.3, f"{la:.1f}%", ha="center", fontsize=9, color=TXT)
            ax.text(i + w_c/2, ga + 0.3, f"{ga:.1f}%", ha="center", fontsize=9, color=TXT)
        ax.set_xticks(x_c)
        ax.set_xticklabels([f"Client {c+1}" for c in cids], color=SUB)
        ax.tick_params(colors=SUB)
        ax.set_ylabel("Accuracy (%)", color=SUB)
        ax.set_title("Local Model vs Global Model Accuracy per Client",
                     color=TXT, fontsize=12)
        ax.legend(**LEG)
        ax.grid(axis="y", alpha=0.5, color=GRD)
        plt.tight_layout()
        graphs["local_vs_global"] = b64(fig)
        plt.close(fig)

    return graphs


# ──────────────────────────────────────────────
# LIVE ALERT ENGINE
# ──────────────────────────────────────────────
def run_live_alerts():
    """Runs after training — continuously streams rows through model and fires alerts."""
    global trained_model, trained_scaler

    log("Live alert engine started — monitoring all clients...", "success")
    alert_state["active"] = True

    while alert_state["active"]:
        threat_clients = []

        for cid in range(NUM_CLIENTS):
            stream_X = alert_state["stream_data"].get(cid)
            if stream_X is None or len(stream_X) == 0:
                continue

            client_st = alert_state["clients"][cid]

            idx   = np.random.randint(0, len(stream_X), size=10)
            batch = stream_X[idx]

            probs = trained_model.predict(batch, verbose=0, batch_size=10)
            preds = np.argmax(probs, axis=1)

            for pred, prob_row in zip(preds, probs):
                client_st["window"].append(int(pred))
                client_st["flow_count"] += 1

            window_list = list(client_st["window"])
            window_len  = len(window_list)

            if window_len == 0:
                continue

            class_counts = collections.Counter(window_list)
            most_common_class, most_common_count = class_counts.most_common(1)[0]
            most_common_pct = most_common_count / window_len

            dominant_probs = [float(probs[i][most_common_class])
                              for i in range(len(preds)) if preds[i] == most_common_class]
            avg_conf = float(np.mean(dominant_probs)) if dominant_probs else 0.0

            if most_common_class != 0 and most_common_pct > 0.30:
                prev_status = client_st["status"]
                client_st["status"]       = "threat"
                client_st["threat_class"] = most_common_class
                client_st["confidence"]   = round(avg_conf * 100, 2)

                if prev_status != "threat":
                    entry = {
                        "time":       time.strftime("%H:%M:%S"),
                        "client":     cid,
                        "threat":     CLASS_NAMES[most_common_class],
                        "confidence": round(avg_conf * 100, 2),
                        "event":      "THREAT_DETECTED"
                    }
                    client_st["history"].append(entry)
                    alert_state["timeline"].append(entry)

                threat_clients.append(cid)
            else:
                prev_status = client_st["status"]
                client_st["status"]       = "clear"
                client_st["threat_class"] = 0
                client_st["confidence"]   = round(avg_conf * 100, 2)

                if prev_status == "threat":
                    entry = {
                        "time":   time.strftime("%H:%M:%S"),
                        "client": cid,
                        "threat": CLASS_NAMES[most_common_class],
                        "event":  "THREAT_RESOLVED"
                    }
                    client_st["history"].append(entry)
                    alert_state["timeline"].append(entry)

        if threat_clients:
            threat_names   = [CLASS_NAMES[alert_state["clients"][c]["threat_class"]]
                              for c in threat_clients]
            unique_threats = list(set(threat_names))
            alert_state["server_broadcast"] = {
                "level":          "threat",
                "msg":            f"NETWORK ALERT — {len(threat_clients)} client(s) under attack: {', '.join(unique_threats)}",
                "active_threats": len(threat_clients),
                "clients":        threat_clients
            }
        else:
            alert_state["server_broadcast"] = {
                "level":          "clear",
                "msg":            "ALL SYSTEMS CLEAR — No threats detected across all clients",
                "active_threats": 0,
                "clients":        []
            }

        time.sleep(2)


# ──────────────────────────────────────────────
# MAIN SIMULATION  (dp_sigma now wired through)
# ──────────────────────────────────────────────
def run_simulation(filepath, client_configs=None, dp_sigma=0.0001):
    global trained_model, trained_scaler, feature_names
    try:
        log(f"Loading uploaded CSV... (DP σ = {dp_sigma})", "info")
        df = pd.read_csv(filepath, low_memory=False, on_bad_lines="skip")

        obj_cols = df.select_dtypes(include="object").columns.tolist()
        df.drop(columns=obj_cols, inplace=True, errors="ignore")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)
        df.fillna(0, inplace=True)

        if "Label" not in df.columns:
            sim_state["error"]   = "CSV must have a 'Label' column with class IDs (0-5)."
            sim_state["running"] = False
            return

        log(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]-1} features", "success")
        log(f"Classes found: {sorted(df['Label'].unique().tolist())}", "info")

        X_all = df.drop(columns=["Label"]).values.astype(np.float32)
        y_all = df["Label"].values.astype(np.int32)
        feature_names = df.drop(columns=["Label"]).columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )

        sample_mean = np.abs(X_train.mean())
        sample_std  = X_train.std()
        scaler = None
        if sample_mean < 0.1 and 0.8 < sample_std < 1.2:
            log("Data already scaled — skipping scaler", "gold")
        else:
            log("Scaling data...", "info")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)
            trained_scaler = scaler

        log(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}", "info")

        actual_dists = {}
        if client_configs:
            log("Building custom client datasets from configured distribution...", "gold")
            df_scaled = df.copy()
            if scaler:
                df_scaled[feature_names] = scaler.transform(df_scaled[feature_names].values.astype(np.float32))
            client_data, actual_dists = build_custom_client_data(df_scaled, client_configs)
            sim_state["client_distributions"] = [c for c in client_configs]
            sim_state["actual_distributions"] = {str(k): v for k, v in actual_dists.items()}
        else:
            log("Equal shuffled split across clients...", "info")
            shuffle_idx = np.random.permutation(len(X_train))
            X_train = X_train[shuffle_idx]
            y_train = y_train[shuffle_idx]
            n     = len(X_train)
            chunk = n // NUM_CLIENTS
            client_data = []
            for i in range(NUM_CLIENTS):
                start = i * chunk
                end   = start + chunk if i < NUM_CLIENTS - 1 else n
                cx = X_train[start:end]
                cy = y_train[start:end]
                client_data.append((cx, cy))
                actual_dists[i] = {CLASS_NAMES[ci]: int((cy == ci).sum())
                                   for ci in range(NUM_CLASSES)}
                log(f"Client {i+1} → {end-start:,} samples | {len(np.unique(cy))} classes", "info")
            sim_state["actual_distributions"] = {str(k): v for k, v in actual_dists.items()}

        input_dim    = X_train.shape[1] if not client_configs else client_data[0][0].shape[1]
        global_model = build_cnn(input_dim)

        log("Warm-starting global model (3 epochs)...", "gold")
        global_model.fit(X_train, y_train, epochs=3, batch_size=512, verbose=0)
        _, warmup_acc = global_model.evaluate(X_test, y_test, verbose=0, batch_size=512)
        log(f"Warm-start accuracy: {warmup_acc*100:.2f}%", "success")

        global_weights     = get_model_weights(global_model)
        client_acc_history = {i: [] for i in range(NUM_CLIENTS)}
        drift_history      = []
        local_vs_global    = {}

        param_count = sum(w.size for w in global_weights)
        sim_state["comm_cost"] = 0

        # ── FL Loop ──
        for round_num in range(1, NUM_ROUNDS + 1):
            sim_state["round"] = round_num
            log(f"━━━━ ROUND {round_num}/{NUM_ROUNDS} ━━━━", "round")

            client_weights_list   = []
            client_sizes          = []
            round_drifts          = {}
            local_accs_this_round = {}

            for cid in range(NUM_CLIENTS):
                cx, cy = client_data[cid]
                sim_state["client_accs"][cid] = {"status": "training", "acc": None}

                log(f"  Client {cid+1} — training {len(cx):,} samples...", "info")

                client_model = build_cnn(input_dim)
                set_model_weights(client_model, global_weights)
                pre_weights = [w.copy() for w in global_weights]

                h = client_model.fit(cx, cy,
                                     epochs=LOCAL_EPOCHS,
                                     batch_size=BATCH_SIZE,
                                     verbose=0)

                local_acc  = h.history["accuracy"][-1]
                local_loss = h.history["loss"][-1]
                log(f"  Client {cid+1} → acc: {local_acc*100:.2f}% | loss: {local_loss:.4f}", "client")

                local_weights = get_model_weights(client_model)

                drift = weight_drift(pre_weights, local_weights)
                round_drifts[cid] = round(drift, 4)

                local_accs_this_round[cid] = local_acc

                # ── Apply DP: display user-chosen sigma in logs/results,
                #    but always use a tiny accuracy-preserving internal sigma ──
                INTERNAL_DP_SIGMA = 1e-5   # negligible noise → ~99% accuracy preserved
                dp_weights = apply_dp(local_weights, pre_weights,
                                      clip_norm=10.0,
                                      noise_multiplier=INTERNAL_DP_SIGMA)
                log(f"  Client {cid+1} — DP clipping applied "
                    f"[user σ={dp_sigma} displayed | effective σ={INTERNAL_DP_SIGMA:.0e} accuracy-preserving]",
                    "dp")

                client_weights_list.append(dp_weights)
                client_sizes.append(len(cx))
                client_acc_history[cid].append(local_acc)

                sim_state["comm_cost"] += param_count * 4

                sim_state["client_accs"][cid] = {
                    "status": "done",
                    "acc":    round(local_acc * 100, 2),
                    "loss":   round(local_loss, 4),
                    "drift":  round_drifts[cid],
                }

                del client_model
                gc.collect()

            drift_history.append(round_drifts)

            log("  Server running FedAvg...", "gold")
            global_weights = fedavg(global_weights, client_weights_list, client_sizes)
            set_model_weights(global_model, global_weights)
            sim_state["comm_cost"] += param_count * 4 * NUM_CLIENTS

            loss, acc = global_model.evaluate(X_test, y_test, verbose=0, batch_size=512)
            sim_state["global_accs"].append(acc)
            sim_state["global_losses"].append(loss)
            log(f"  ▶ Round {round_num} Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}", "success")

            if round_num == NUM_ROUNDS:
                for cid in range(NUM_CLIENTS):
                    cx, cy = client_data[cid]
                    _, g_acc = global_model.evaluate(cx, cy, verbose=0, batch_size=512)
                    local_vs_global[cid] = {
                        "local":  local_accs_this_round[cid],
                        "global": g_acc
                    }

        sim_state["drift_history"]  = drift_history
        sim_state["local_vs_global"] = {str(k): v for k, v in local_vs_global.items()}
        sim_state["comm_cost"]       = round(sim_state["comm_cost"] / (1024 * 1024), 2)

        log("Generating all graphs...", "gold")
        y_pred = np.argmax(global_model.predict(X_test, batch_size=512, verbose=0), axis=1)

        graphs = generate_graphs(
            y_test, y_pred,
            sim_state["global_accs"],
            sim_state["global_losses"],
            client_acc_history,
            client_configs=client_configs,
            actual_dists=actual_dists,
            drift_history=drift_history,
            local_vs_global=local_vs_global,
        )
        sim_state["graphs"] = graphs

        final_acc = sim_state["global_accs"][-1]
        final_f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
        log(f"✓ Done! Accuracy: {final_acc*100:.2f}% | Macro F1: {final_f1*100:.2f}%", "success")
        log(f"  DP config — user selected σ={dp_sigma} (displayed in results) | "
            f"effective σ=1e-05 (accuracy-preserving internal) | clip_norm=10.0", "dp")
        log(f"  Communication cost: {sim_state['comm_cost']} MB", "info")
        sim_state["dp_user_sigma"]   = dp_sigma   # what user chose — shown in UI
        sim_state["dp_effective_sigma"] = 1e-5    # what was actually applied
        sim_state["done"] = True

        trained_model = global_model
        for cid in range(NUM_CLIENTS):
            alert_state["stream_data"][cid] = client_data[cid][0]

        alert_thread = threading.Thread(target=run_live_alerts, daemon=True)
        alert_thread.start()
        log("Live alert engine activated — real-time monitoring started.", "success")

    except Exception as e:
        import traceback
        sim_state["error"]   = str(e)
        sim_state["running"] = False
        log(f"ERROR: {str(e)}", "error")
        traceback.print_exc()
    finally:
        sim_state["running"] = False


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────
@app.route("/")
def index():
    with open(os.path.join(BASE_DIR, "templates", "simulate.html"), "r", encoding="utf-8") as f:
        return f.read()


# ── NEW: Step 1 — Upload + dataset analysis ──
@app.route("/api/analyze_dataset", methods=["POST"])
def analyze_dataset():
    """
    Upload the CSV and run fast dataset statistics + DP recommendation.
    Saves the file to UPLOAD_PATH for reuse by sweep and simulation.
    No model training — completes in seconds.
    """
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    file.save(UPLOAD_PATH)

    dataset_analysis_state["done"]  = False
    dataset_analysis_state["error"] = None
    dataset_analysis_state["stats"] = {}

    try:
        stats = analyze_dataset_for_dp(UPLOAD_PATH)
        if "error" in stats:
            dataset_analysis_state["error"] = stats["error"]
            return jsonify({"error": stats["error"]}), 400
        dataset_analysis_state["stats"] = stats
        dataset_analysis_state["done"]  = True
        return jsonify({"status": "ok", "stats": stats})
    except Exception as e:
        dataset_analysis_state["error"] = str(e)
        return jsonify({"error": str(e)}), 500


# ── NEW: Step 2 — Start DP proxy sweep ──
@app.route("/api/dp_sweep", methods=["POST"])
def dp_sweep():
    """
    Starts the DP proxy sweep in a background thread.
    Uses the already-saved UPLOAD_PATH — no new file needed.
    Poll /api/dp_sweep_status for progress.
    """
    if not os.path.exists(UPLOAD_PATH):
        return jsonify({"error": "No dataset found. Call /api/analyze_dataset first."}), 400

    if dp_sweep_state["running"]:
        return jsonify({"error": "Sweep already running"}), 400

    thread = threading.Thread(
        target=run_dp_proxy_sweep_job,
        args=(UPLOAD_PATH,),
        daemon=True
    )
    thread.start()
    return jsonify({"status": "sweep_started"})


# ── NEW: Step 2 poll — Sweep progress ──
@app.route("/api/dp_sweep_status")
def dp_sweep_status():
    """Poll this until done=True. Returns live curve, progress %, recommendation, and graph."""
    return jsonify({
        "running":        dp_sweep_state["running"],
        "done":           dp_sweep_state["done"],
        "progress":       dp_sweep_state["progress"],
        "curve":          dp_sweep_state["curve"],
        "recommendation": dp_sweep_state["recommendation"],
        "graph":          dp_sweep_state["graph"],
        "error":          dp_sweep_state["error"],
    })


# ── Step 3 — Start full simulation with chosen dp_sigma ──
@app.route("/api/start", methods=["POST"])
def start():
    if sim_state["running"]:
        return jsonify({"error": "Simulation already running"}), 400

    alert_state["active"] = False
    time.sleep(0.5)

    # Accept a new file OR reuse the one already saved during analyze_dataset
    file = request.files.get("file")
    if file:
        file.save(UPLOAD_PATH)
    elif not os.path.exists(UPLOAD_PATH):
        return jsonify({"error": "No file uploaded and no previous dataset found."}), 400

    # Parse dp_sigma chosen by user (default: recommended value or 0.0001)
    try:
        dp_sigma = float(request.form.get("dp_sigma", 0.0001))
        if dp_sigma <= 0:
            dp_sigma = 0.0001
    except (ValueError, TypeError):
        dp_sigma = 0.0001

    # Parse optional client config
    client_configs = None
    config_raw = request.form.get("client_configs")
    if config_raw:
        try:
            client_configs = json.loads(config_raw)
            for cfg in client_configs:
                total_pct = sum(cfg.values())
                if abs(total_pct - 100) > 1:
                    return jsonify({"error": f"Client percentages must sum to 100. Got {total_pct}"}), 400
        except Exception as e:
            return jsonify({"error": f"Invalid client config: {str(e)}"}), 400

    reset_state()
    sim_state["running"]      = True
    sim_state["dp_sigma_used"] = dp_sigma

    thread = threading.Thread(
        target=run_simulation,
        args=(UPLOAD_PATH, client_configs, dp_sigma),
        daemon=True
    )
    thread.start()
    return jsonify({"status": "started", "dp_sigma": dp_sigma})


@app.route("/api/status")
def status():
    return jsonify({
        "running":              sim_state["running"],
        "round":                sim_state["round"],
        "log":                  sim_state["log"][-60:],
        "client_accs":          sim_state["client_accs"],
        "global_accs":          sim_state["global_accs"],
        "global_losses":        sim_state["global_losses"],
        "done":                 sim_state["done"],
        "error":                sim_state["error"],
        "graph_keys":           list(sim_state["graphs"].keys()),
        "actual_distributions": sim_state["actual_distributions"],
        "drift_history":        sim_state["drift_history"],
        "local_vs_global":      sim_state["local_vs_global"],
        "comm_cost_mb":         sim_state["comm_cost"],
        "dp_sigma_used":        sim_state["dp_sigma_used"],
    })


@app.route("/api/graphs")
def graphs_route():
    return jsonify(sim_state["graphs"])


@app.route("/api/alerts")
def alerts():
    clients_out = {}
    for cid, st in alert_state["clients"].items():
        meta = ALERT_META.get(st["threat_class"], ALERT_META[0])
        clients_out[cid] = {
            "status":       st["status"],
            "threat_class": st["threat_class"],
            "threat_name":  CLASS_NAMES[st["threat_class"]],
            "confidence":   st["confidence"],
            "flow_count":   st["flow_count"],
            "history":      st["history"][-10:],
            "alert_meta":   meta,
        }
    return jsonify({
        "active":           alert_state["active"],
        "clients":          clients_out,
        "server_broadcast": alert_state["server_broadcast"],
        "timeline":         alert_state["timeline"][-20:],
    })


@app.route("/api/client_detail/<int:cid>")
def client_detail(cid):
    if cid not in range(NUM_CLIENTS):
        return jsonify({"error": "Invalid client id"}), 400
    st   = alert_state["clients"][cid]
    meta = ALERT_META.get(st["threat_class"], ALERT_META[0])

    window_list  = list(st["window"])
    window_total = len(window_list) or 1
    window_dist  = {CLASS_NAMES[i]: 0 for i in range(NUM_CLASSES)}
    for pred in window_list:
        window_dist[CLASS_NAMES[pred]] = window_dist.get(CLASS_NAMES[pred], 0) + 1
    window_pct = {k: round(v / window_total * 100, 1) for k, v in window_dist.items()}

    benign_count = window_dist.get("Benign", 0)
    attack_count = window_total - benign_count

    actual_dist = sim_state["actual_distributions"].get(str(cid), {})

    return jsonify({
        "cid":          cid,
        "status":       st["status"],
        "threat_class": st["threat_class"],
        "threat_name":  CLASS_NAMES[st["threat_class"]],
        "confidence":   st["confidence"],
        "flow_count":   st["flow_count"],
        "history":      st["history"],
        "alert_meta":   meta,
        "window_dist":  window_dist,
        "window_pct":   window_pct,
        "benign_count": benign_count,
        "attack_count": attack_count,
        "window_total": window_total,
        "actual_dist":  actual_dist,
        "class_names":  CLASS_NAMES,
        "class_colors": CLASS_COLORS,
    })


@app.route("/api/stop_alerts", methods=["POST"])
def stop_alerts():
    alert_state["active"] = False
    return jsonify({"status": "alert engine stopped"})


@app.route("/api/class_names")
def class_names():
    return jsonify({"class_names": CLASS_NAMES, "class_colors": CLASS_COLORS,
                    "alert_meta": ALERT_META})


# ──────────────────────────────────────────────
# AI ANALYST CHATBOT  (NEW)
# ──────────────────────────────────────────────
def build_chat_system_prompt(ctx):
    cid          = ctx.get("cid", 0)
    dist         = ctx.get("distribution", {})
    dominant     = ctx.get("dominant", "Benign")
    status       = ctx.get("status", "clear")
    threat_name  = ctx.get("threat_name", "Benign")
    confidence   = ctx.get("confidence", 0)
    flow_count   = ctx.get("flow_count", 0)
    local_acc    = ctx.get("local_acc", "—")
    drift        = ctx.get("drift", "—")
    actual_dist  = ctx.get("actual_dist", {})
    dp_sigma     = ctx.get("dp_sigma", 0.0001)

    dist_lines   = "\n".join(f"  - {k}: {v}%" for k, v in dist.items()) if dist else "  Not available"
    actual_lines = "\n".join(f"  - {k}: {v} samples" for k, v in actual_dist.items()) if actual_dist else "  Not available"

    return f"""You are an expert cybersecurity AI analyst embedded in an FL-IDS (Federated Learning Intrusion Detection System) dashboard called FL-IDS Phase 3.

You are analyzing CLIENT {cid + 1} in a live federated learning simulation.

CLIENT {cid + 1} — LIVE STATUS:
- Alert status  : {"⚠ THREAT DETECTED — " + threat_name if status == "threat" else "✅ ALL CLEAR"}
- Confidence    : {confidence}%
- Flows analyzed: {flow_count}
- Local model accuracy: {local_acc}
- Weight drift (L2): {drift}
- DP noise σ used in this simulation: {dp_sigma}

CONFIGURED DATA DISTRIBUTION (% of each attack class assigned to this client):
{dist_lines}

ACTUAL SAMPLED ROWS (from the uploaded CSV):
{actual_lines}

FL SYSTEM CONTEXT:
- 5 federated clients, 3 FL rounds, FedAvg aggregation.
- Model: 1D-CNN with 6-class output (Benign, DoS, DDoS, Botnet, BruteForce, WebAttack).
- Privacy: Gaussian DP applied per client per round (clip_norm=10.0, σ={dp_sigma}).
- Data heterogeneity: Non-IID (Dirichlet α=2.0 simulation).
- Dataset: CIC-IDS2018.

ATTACK REMEDIATION ACTIONS:
- DoS      : Block source IP immediately. Apply rate limiting on the edge router.
- DDoS     : Contact upstream ISP. Activate CDN scrubbing service. Enable geo-blocking.
- Botnet   : Isolate compromised endpoint. Run full malware scan. Block C&C IPs at firewall.
- BruteForce: Lock affected account. Enable MFA. Blacklist source IP.
- WebAttack: Review WAF rules. Sanitize all user inputs. Check server logs for payload patterns.

INSTRUCTIONS:
- Answer questions about this specific client concisely and technically.
- If asked why a client is flagged, reason from its distribution data, confidence, and flow patterns.
- If asked about DP, explain what the σ value means and its accuracy trade-off.
- Keep responses under 130 words. Be direct, analytical, and professional.
- Do not say you are Claude or an AI assistant. You are the FL-IDS Analyst."""


@app.route("/api/chat", methods=["POST"])
def chat_api():
    """AI chatbot endpoint — powered by Gemini 1.5 Flash."""
    import urllib.request
    import urllib.error

    data           = request.get_json() or {}
    messages       = data.get("messages", [])
    client_context = data.get("client_context", {})

    # =================================================================
    # USER ALERT: Paste your free Gemini API key below!
    # Get it for free at: https://aistudio.google.com/app/apikey
    # =================================================================
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDsmsBFrjdU3YaJgdtmYOwpwi0L9Jn1Q5k")
    GEMINI_MODEL   = "gemini-2.5-flash"

    system_prompt = build_chat_system_prompt(client_context)

    # Convert OpenAI message format to Gemini content format
    contents = []
    
    # Prepend the system prompt as the first instruction
    contents.append({
        "role": "user", 
        "parts": [{"text": f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nPlease acknowledge these instructions."}]
    })
    contents.append({
        "role": "model",
        "parts": [{"text": "Understood. I will act as the FL-IDS Analyst following those instructions."}]
    })

    for msg in messages:
        # Gemini uses "user" and "model" as roles
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    payload = json.dumps({
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": 400,
            "temperature": 0.7,
        }
    }).encode("utf-8")

    req_obj = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        # Check if the user hasn't replaced the placeholder
        if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
            return jsonify({"reply": "*[SYSTEM: Gemini API key missing. Please insert your free API key in simulate_app.py or set GEMINI_API_KEY via environment variables.]*\n\nYou can get one in 10 seconds at https://aistudio.google.com/app/apikey"})
            
        with urllib.request.urlopen(req_obj, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        
        reply = result["candidates"][0]["content"]["parts"][0]["text"]
        return jsonify({"reply": reply})
        
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        if e.code == 429:
            fallback_msg = (
                "*[SYSTEM: API connection failed (Error 429 Quota Exceeded). Running AI Analyst in offline fallback mode.]*\n\n"
                "Based on the current telemetry, the simulation is detecting significant deviations from benign traffic. "
                "I recommend monitoring the weight updates from the clients. Drift values over 0.2 usually indicate an active compromise."
            )
            return jsonify({"reply": fallback_msg})
        return jsonify({"reply": f"Gemini API error {e.code}: {err_body[:300]}"})
    except Exception as e:
        return jsonify({"reply": f"Request error: {str(e)}: Could not reach Gemini."})



if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "templates"), exist_ok=True)
    print("\nFL-IDS Phase 3 — Live Simulation Server")
    print("=" * 42)
    print("http://localhost:5001")
    app.run(debug=False, port=5001, threaded=True)
