import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data_processed")
STATIC_DIR = os.path.join(BASE_DIR, "static")

CLASS_NAMES  = ["Benign", "DoS", "DDoS", "Botnet", "BruteForce", "WebAttack"]
CLASS_COLORS = ["#27AE60", "#E67E22", "#C0392B", "#8E44AD", "#2980B9", "#F39C12"]

print("Loading model...")
model = tf.keras.models.load_model(os.path.join(DATA_DIR, "fl_ids_phase2_global_model.keras"))
print("Model loaded.")

print("Fitting scaler on test data...")
X_test = np.load(os.path.join(DATA_DIR, "X_test_phase2.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test_phase2.npy"))
scaler = StandardScaler()
scaler.fit(X_test)
print("Scaler ready.")

feature_names = pd.read_csv(
    os.path.join(DATA_DIR, "selected_features_phase2.csv")
).iloc[:, 0].tolist()

with open(os.path.join(DATA_DIR, "phase2_results.json")) as f:
    phase2_results = json.load(f)

print(f"Features expected: {len(feature_names)}")


@app.route("/")
def index():
    with open(os.path.join(BASE_DIR, "templates", "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.route("/static/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(DATA_DIR, filename)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        df = pd.read_csv(file, low_memory=False)

        obj_cols = df.select_dtypes(include="object").columns.tolist()
        df.drop(columns=obj_cols, inplace=True, errors="ignore")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        available = [f for f in feature_names if f in df.columns]
        missing   = [f for f in feature_names if f not in df.columns]

        if len(available) < 10:
            return jsonify({
                "error": f"CSV has too few matching features. Found {len(available)}/96."
            }), 400

        for m in missing:
            df[m] = 0.0
        df = df[feature_names]

        X = scaler.transform(df.values)
        probs = model.predict(X, batch_size=256, verbose=0)
        preds = np.argmax(probs, axis=1)

        results = []
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            results.append({
                "row":        i + 1,
                "prediction": CLASS_NAMES[pred],
                "confidence": round(float(prob[pred]) * 100, 2),
                "class_id":   int(pred),
                "all_probs":  {CLASS_NAMES[j]: round(float(prob[j]) * 100, 2)
                               for j in range(len(CLASS_NAMES))}
            })

        summary = {name: 0 for name in CLASS_NAMES}
        for r in results:
            summary[r["prediction"]] += 1

        return jsonify({
            "total":   len(results),
            "results": results[:200],
            "summary": summary,
            "colors":  CLASS_COLORS
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dp_demo", methods=["POST"])
def dp_demo():
    try:
        data       = request.json
        weights    = np.array(data["weights"], dtype=float)
        clip_norm  = float(data.get("clip_norm", 10.0))
        noise_mult = float(data.get("noise_multiplier", 0.0001))

        norm = np.linalg.norm(weights)
        clipped = weights * (clip_norm / norm) if norm > clip_norm else weights.copy()
        noise   = np.random.normal(0, noise_mult * clip_norm, size=clipped.shape)
        dp_weights = clipped + noise

        return jsonify({
            "original":    weights.tolist(),
            "clipped":     clipped.tolist(),
            "dp_weights":  dp_weights.tolist(),
            "noise":       noise.tolist(),
            "original_norm": round(float(norm), 4),
            "clipped_norm":  round(float(np.linalg.norm(clipped)), 4),
            "noise_std":     round(float(noise.std()), 6),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/results")
def results():
    return jsonify(phase2_results)


@app.route("/api/simulate_fl")
def simulate_fl():
    rounds = []
    for i, (acc, loss) in enumerate(zip(
        phase2_results["round_accuracies"],
        phase2_results["round_losses"]
    )):
        rounds.append({
            "round":    i + 1,
            "accuracy": round(acc * 100, 2),
            "loss":     round(loss, 4),
            "clients":  [
                {"id": c + 1, "acc": round(acc * 100 - (c * 0.05), 2)}
                for c in range(5)
            ]
        })
    return jsonify({
        "rounds":      rounds,
        "num_clients": 5,
        "num_rounds":  5,
        "final_acc":   round(phase2_results["final_accuracy"] * 100, 2),
        "macro_f1":    round(phase2_results["macro_f1"] * 100, 2),
    })


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "templates"), exist_ok=True)
    print("\nFL-IDS Phase 2 Dashboard")
    print("=" * 30)
    print("Starting server at http://localhost:5000")
    app.run(debug=False, port=5000)