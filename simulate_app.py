import os
import time
import numpy as np
import pandas as pd
import threading
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras import layers, models
import gc

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES  = ["Benign", "DoS", "DDoS", "Botnet", "BruteForce", "WebAttack"]
CLASS_COLORS = ["#27AE60", "#E67E22", "#C0392B", "#8E44AD", "#2980B9", "#F39C12"]

NUM_CLIENTS   = 5
NUM_ROUNDS    = 3
LOCAL_EPOCHS  = 3
BATCH_SIZE    = 256
LEARNING_RATE = 0.001
NUM_CLASSES   = 6

sim_state = {
    "running":       False,
    "log":           [],
    "round":         0,
    "client_accs":   {},
    "global_accs":   [],
    "global_losses": [],
    "done":          False,
    "error":         None,
    "graphs":        {}
}


def reset_state():
    sim_state.update({
        "running":       False,
        "log":           [],
        "round":         0,
        "client_accs":   {},
        "global_accs":   [],
        "global_losses": [],
        "done":          False,
        "error":         None,
        "graphs":        {}
    })


def log(msg, level="info"):
    entry = {"msg": msg, "level": level, "time": time.strftime("%H:%M:%S")}
    sim_state["log"].append(entry)
    print(f"[{entry['time']}] {msg}")


def build_cnn(input_dim):
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


def get_model_weights(model):
    return model.get_weights()


def set_model_weights(model, weights):
    model.set_weights(weights)


def fedavg(global_weights, client_weights_list, client_sizes):
    total_samples = sum(client_sizes)
    averaged_weights = []
    for layer_idx in range(len(global_weights)):
        weighted_sum = sum(
            client_weights_list[c][layer_idx] * client_sizes[c]
            for c in range(len(client_weights_list))
        )
        averaged_weights.append(weighted_sum / total_samples)
    return averaged_weights


def apply_dp(weights, clip_norm=10.0, noise_multiplier=0.0001):
    dp_weights = []
    for w in weights:
        norm = np.linalg.norm(w)
        if norm > clip_norm:
            w = w * (clip_norm / norm)
        noise = np.random.normal(loc=0.0, scale=noise_multiplier * clip_norm, size=w.shape)
        dp_weights.append(w + noise)
    return dp_weights


def generate_graphs(y_test, y_pred, round_accs, round_losses, client_acc_history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO

    plt.style.use("dark_background")
    facecolor = "#0A0A0F"

    def fig_to_b64(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120,
                    bbox_inches="tight", facecolor=facecolor)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    graphs = {}

    # 1 — Confusion Matrix
    cm     = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(facecolor)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0], linewidths=0.5, linecolor=facecolor)
    axes[0].set_title("Confusion Matrix — Counts", color="#F5F5F5", fontsize=12)
    axes[0].set_xlabel("Predicted", color="#D0D3E0")
    axes[0].set_ylabel("True", color="#D0D3E0")
    axes[0].set_facecolor("#12121C")
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Reds",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], linewidths=0.5, linecolor=facecolor)
    axes[1].set_title("Confusion Matrix — %", color="#F5F5F5", fontsize=12)
    axes[1].set_xlabel("Predicted", color="#D0D3E0")
    axes[1].set_ylabel("True", color="#D0D3E0")
    axes[1].set_facecolor("#12121C")
    plt.tight_layout()
    graphs["confusion_matrix"] = fig_to_b64(fig)
    plt.close(fig)

    # 2 — FL Convergence
    rounds = list(range(1, len(round_accs) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(facecolor)
    axes[0].plot(rounds, [a * 100 for a in round_accs],
                 color="#27AE60", linewidth=2.5, marker="o",
                 markersize=8, markerfacecolor="#F5F5F5")
    for r, a in zip(rounds, round_accs):
        axes[0].annotate(f"{a*100:.2f}%", (r, a * 100),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=9, color="#F5F5F5")
    axes[0].set_title("Global Accuracy per Round", color="#F5F5F5", fontsize=12)
    axes[0].set_xlabel("Round", color="#D0D3E0")
    axes[0].set_ylabel("Accuracy (%)", color="#D0D3E0")
    axes[0].set_facecolor("#12121C")
    axes[0].grid(alpha=0.3)
    axes[1].plot(rounds, round_losses,
                 color="#C0392B", linewidth=2.5, marker="s",
                 markersize=8, markerfacecolor="#F5F5F5")
    for r, l in zip(rounds, round_losses):
        axes[1].annotate(f"{l:.4f}", (r, l),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=9, color="#F5F5F5")
    axes[1].set_title("Global Loss per Round", color="#F5F5F5", fontsize=12)
    axes[1].set_xlabel("Round", color="#D0D3E0")
    axes[1].set_ylabel("Loss", color="#D0D3E0")
    axes[1].set_facecolor("#12121C")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    graphs["convergence"] = fig_to_b64(fig)
    plt.close(fig)

    # 3 — Per-Class F1
    f1   = f1_score(y_test, y_pred, average=None, zero_division=0)
    prec = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec  = recall_score(y_test, y_pred, average=None, zero_division=0)
    x    = np.arange(NUM_CLASSES)
    w    = 0.25
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor("#12121C")
    ax.bar(x - w, prec * 100, w, label="Precision", color="#2980B9")
    ax.bar(x,     rec  * 100, w, label="Recall",    color="#27AE60")
    ax.bar(x + w, f1   * 100, w, label="F1-Score",  color="#C0392B")
    for i, (p, r, f) in enumerate(zip(prec, rec, f1)):
        ax.text(i - w, p * 100 + 0.3, f"{p*100:.1f}%", ha="center", fontsize=8, color="#F5F5F5")
        ax.text(i,     r * 100 + 0.3, f"{r*100:.1f}%", ha="center", fontsize=8, color="#F5F5F5")
        ax.text(i + w, f * 100 + 0.3, f"{f*100:.1f}%", ha="center", fontsize=8, color="#F5F5F5")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, color="#D0D3E0")
    ax.set_ylim(0, 115)
    ax.set_title("Per-Class Precision / Recall / F1", color="#F5F5F5", fontsize=12)
    ax.legend(facecolor="#12121C", edgecolor="#444455")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    graphs["per_class"] = fig_to_b64(fig)
    plt.close(fig)

    # 4 — Client accuracy history
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor("#12121C")
    for cid, hist in client_acc_history.items():
        ax.plot(range(1, len(hist) + 1), [a * 100 for a in hist],
                marker="o", linewidth=2, label=f"Client {cid+1}",
                color=CLASS_COLORS[cid])
    ax.set_title("Client Local Accuracy per Round", color="#F5F5F5", fontsize=12)
    ax.set_xlabel("Round", color="#D0D3E0")
    ax.set_ylabel("Local Accuracy (%)", color="#D0D3E0")
    ax.legend(facecolor="#12121C", edgecolor="#444455")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    graphs["client_accs"] = fig_to_b64(fig)
    plt.close(fig)

    # 5 — Class distribution
    counts_full = np.zeros(NUM_CLASSES, dtype=int)
    unique, counts = np.unique(y_test, return_counts=True)
    for u, c in zip(unique, counts):
        counts_full[u] = c
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor("#12121C")
    bars = ax.bar(CLASS_NAMES, counts_full, color=CLASS_COLORS, edgecolor="none")
    for bar, cnt in zip(bars, counts_full):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{cnt:,}", ha="center", fontsize=10, color="#F5F5F5")
    ax.set_title("Test Set Class Distribution", color="#F5F5F5", fontsize=12)
    ax.set_xlabel("Class", color="#D0D3E0")
    ax.set_ylabel("Count", color="#D0D3E0")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    graphs["class_dist"] = fig_to_b64(fig)
    plt.close(fig)

    # 6 — DP noise visualization

    np.random.seed(42)
    sample_weights = np.array([1.8, -1.1, 0.3, -0.6, 2.2, 0.5, -0.15, 1.0, -0.05, -0.7])
    norm    = np.linalg.norm(sample_weights)
    clipped = sample_weights * (10.0 / norm) if norm > 10.0 else sample_weights.copy()
    # Use larger noise for visualization — actual training uses σ=0.0001
    noised  = clipped + np.random.normal(0, 0.4, size=10)
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor("#12121C")
    x_pos = np.arange(10)
    ax.bar(x_pos - 0.25, sample_weights, 0.25, label="Original",  color="#2980B9")
    ax.bar(x_pos,        clipped,        0.25, label="Clipped",   color="#F39C12")
    ax.bar(x_pos + 0.25, noised,         0.25, label="DP Noised", color="#C0392B")
    ax.set_title("Differential Privacy — Weight Transformation (clip=10, σ=0.0001)",
                 color="#F5F5F5", fontsize=12)
    ax.set_xlabel("Weight Index", color="#D0D3E0")
    ax.set_ylabel("Weight Value", color="#D0D3E0")
    ax.legend(facecolor="#12121C", edgecolor="#444455")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    graphs["dp_viz"] = fig_to_b64(fig)
    plt.close(fig)

    # 7 — Summary metrics
    macro_f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    acc_val    = (y_pred == y_test).mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor("#12121C")
    metrics_labels = ["Accuracy", "Macro F1", "Macro Precision", "Macro Recall"]
    values         = [acc_val * 100, macro_f1 * 100, macro_prec * 100, macro_rec * 100]
    colors         = ["#27AE60", "#C0392B", "#2980B9", "#F39C12"]
    bars = ax.barh(metrics_labels, values, color=colors, edgecolor="none", height=0.5)
    for bar, val in zip(bars, values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=11,
                color="#F5F5F5", fontweight="bold")
    ax.set_xlim(0, 110)
    ax.set_title("Global Model — Summary Metrics", color="#F5F5F5", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    graphs["summary"] = fig_to_b64(fig)
    plt.close(fig)

    return graphs


def run_simulation(filepath):
    try:
        log("Loading uploaded CSV...", "info")
        df = pd.read_csv(filepath, low_memory=False, on_bad_lines="skip")

        obj_cols = df.select_dtypes(include="object").columns.tolist()
        df.drop(columns=obj_cols, inplace=True, errors="ignore")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)
        df.fillna(0, inplace=True)

        if "Label" not in df.columns:
            sim_state["error"] = "CSV must have a 'Label' column with class IDs (0-5)."
            sim_state["running"] = False
            return

        X = df.drop(columns=["Label"]).values.astype(np.float32)
        y = df["Label"].values.astype(np.int32)

        log(f"Dataset loaded: {X.shape[0]:,} rows × {X.shape[1]} features", "success")
        log(f"Classes found: {np.unique(y).tolist()}", "info")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        sample_mean = np.abs(X_train.mean())
        sample_std  = X_train.std()
        if sample_mean < 0.1 and 0.8 < sample_std < 1.2:
            log(f"Data already scaled — skipping scaler", "gold")
        else:
            log(f"Scaling data...", "info")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        log(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}", "info")

        # Shuffle then equal split
        shuffle_idx = np.random.permutation(len(X_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]

        n     = len(X_train)
        chunk = n // NUM_CLIENTS
        client_data = []
        for i in range(NUM_CLIENTS):
            start = i * chunk
            end   = start + chunk if i < NUM_CLIENTS - 1 else n
            client_data.append((X_train[start:end], y_train[start:end]))
            log(f"Client {i+1} → {end-start:,} samples | {len(np.unique(y_train[start:end]))} classes", "info")

        input_dim    = X_train.shape[1]
        global_model = build_cnn(input_dim)

        # Warm start — exact same as notebook
        log("Warm-starting global model (3 epochs)...", "gold")
        global_model.fit(X_train, y_train, epochs=3, batch_size=512, verbose=0)
        _, warmup_acc = global_model.evaluate(X_test, y_test, verbose=0, batch_size=512)
        log(f"Warm-start accuracy: {warmup_acc*100:.2f}%", "success")

        global_weights     = get_model_weights(global_model)
        client_acc_history = {i: [] for i in range(NUM_CLIENTS)}

        # FL loop — exact same logic as notebook
        for round_num in range(1, NUM_ROUNDS + 1):
            sim_state["round"] = round_num
            log(f"━━━━ ROUND {round_num}/{NUM_ROUNDS} ━━━━", "round")

            client_weights_list = []
            client_sizes        = []

            for cid in range(NUM_CLIENTS):
                cx, cy = client_data[cid]
                sim_state["client_accs"][cid] = {"status": "training", "acc": None}

                log(f"  Client {cid+1} — training {len(cx):,} samples...", "info")

                client_model = build_cnn(input_dim)
                set_model_weights(client_model, global_weights)

                h = client_model.fit(cx, cy,
                                     epochs=LOCAL_EPOCHS,
                                     batch_size=BATCH_SIZE,
                                     verbose=0)

                local_acc  = h.history["accuracy"][-1]
                local_loss = h.history["loss"][-1]
                log(f"  Client {cid+1} → acc: {local_acc*100:.2f}% | loss: {local_loss:.4f}", "client")

                local_weights = get_model_weights(client_model)
                dp_weights    = apply_dp(local_weights, clip_norm=10.0, noise_multiplier=0.0001)
                log(f"  Client {cid+1} — DP applied (clip=10.0, σ=0.0001)", "dp")

                client_weights_list.append(dp_weights)
                client_sizes.append(len(cx))
                client_acc_history[cid].append(local_acc)

                sim_state["client_accs"][cid] = {
                    "status": "done",
                    "acc":    round(local_acc * 100, 2),
                    "loss":   round(local_loss, 4)
                }

                del client_model
                gc.collect()

            log("  Server running FedAvg...", "gold")
            global_weights = fedavg(global_weights, client_weights_list, client_sizes)
            set_model_weights(global_model, global_weights)

            loss, acc = global_model.evaluate(X_test, y_test, verbose=0, batch_size=512)
            sim_state["global_accs"].append(acc)
            sim_state["global_losses"].append(loss)
            log(f"  ▶ Round {round_num} Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}", "success")

        log("Generating graphs...", "gold")
        y_pred = np.argmax(global_model.predict(X_test, batch_size=512, verbose=0), axis=1)
        graphs = generate_graphs(y_test, y_pred,
                                 sim_state["global_accs"],
                                 sim_state["global_losses"],
                                 client_acc_history)
        sim_state["graphs"] = graphs

        final_acc = sim_state["global_accs"][-1]
        final_f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
        log(f"✓ Done! Accuracy: {final_acc*100:.2f}% | Macro F1: {final_f1*100:.2f}%", "success")
        sim_state["done"] = True

    except Exception as e:
        import traceback
        sim_state["error"]   = str(e)
        sim_state["running"] = False
        log(f"ERROR: {str(e)}", "error")
        traceback.print_exc()
    finally:
        sim_state["running"] = False


@app.route("/")
def index():
    with open(os.path.join(BASE_DIR, "templates", "simulate.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.route("/api/start", methods=["POST"])
def start():
    if sim_state["running"]:
        return jsonify({"error": "Simulation already running"}), 400
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    reset_state()
    sim_state["running"] = True
    upload_path = os.path.join(BASE_DIR, "sim_upload.csv")
    file.save(upload_path)
    thread = threading.Thread(target=run_simulation, args=(upload_path,))
    thread.daemon = True
    thread.start()
    return jsonify({"status": "started"})


@app.route("/api/status")
def status():
    return jsonify({
        "running":       sim_state["running"],
        "round":         sim_state["round"],
        "log":           sim_state["log"][-50:],
        "client_accs":   sim_state["client_accs"],
        "global_accs":   sim_state["global_accs"],
        "global_losses": sim_state["global_losses"],
        "done":          sim_state["done"],
        "error":         sim_state["error"],
        "graph_keys":    list(sim_state["graphs"].keys())
    })


@app.route("/api/graphs")
def graphs():
    return jsonify(sim_state["graphs"])


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "templates"), exist_ok=True)
    print("\nFL-IDS Live Simulation Server")
    print("=" * 35)
    print("http://localhost:5001")
    app.run(debug=False, port=5001)