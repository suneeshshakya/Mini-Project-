from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Load Models ───────────────────────────────────────────
model        = joblib.load(os.path.join(BASE_DIR, "dropout_model_v2.pkl"))
scaler       = joblib.load(os.path.join(BASE_DIR, "scaler_v2.pkl"))
le_target    = joblib.load(os.path.join(BASE_DIR, "label_encoder_v2.pkl"))
le_gender    = joblib.load(os.path.join(BASE_DIR, "gender_encoder_v2.pkl"))
feature_cols = joblib.load(os.path.join(BASE_DIR, "feature_cols_v2.pkl"))

# ─── SQLite Setup ──────────────────────────────────────────
DB_PATH = os.path.join(BASE_DIR, "aura.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            gender TEXT,
            academic_year INTEGER,
            study_hours REAL,
            exam_pressure REAL,
            academic_performance REAL,
            stress_level REAL,
            anxiety_score REAL,
            depression_score REAL,
            sleep_hours REAL,
            physical_activity REAL,
            social_support REAL,
            screen_time REAL,
            internet_usage REAL,
            financial_stress REAL,
            family_expectation REAL,
            risk_level TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ─── Helper ────────────────────────────────────────────────
def get_advice(risk):
    advice = {
        "High": [
            "Immediate counseling session recommended",
            "Reduce exam pressure with a structured study plan",
            "Ensure minimum 7 hours of sleep per night",
            "Limit screen time to under 4 hours per day",
            "Connect student with mental health support resources"
        ],
        "Medium": [
            "Schedule a wellness check-in this week",
            "Encourage participation in physical activities",
            "Suggest peer study groups to improve social support",
            "Monitor stress levels over the next 2 weeks"
        ],
        "Low": [
            "Student is on a healthy track — maintain routine monitoring",
            "Encourage continued engagement with academics and social activities"
        ]
    }
    return advice.get(risk, [])

# ─── Routes ────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "AURA.ai Backend",
        "version": "2.0",
        "endpoints": ["/predict", "/history", "/stats", "/health"]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "GradientBoostingClassifier", "accuracy": "84.25%"})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        if request.method == "POST":
            body = request.get_json(force=True)
            get = lambda k, default=None: body.get(k, default)
        else:
            get = lambda k, default=None: request.args.get(k, default)

        # ── Collect Inputs ──
        name                 = get("name", "Unknown")
        age                  = int(get("age", 20))
        gender_raw           = get("gender", "Male")
        academic_year        = int(get("academic_year", 2))
        study_hours          = float(get("study_hours", 5))
        exam_pressure        = float(get("exam_pressure", 5))
        academic_performance = float(get("academic_performance", 65))
        stress_level         = float(get("stress_level", 5))
        anxiety_score        = float(get("anxiety_score", 3))
        depression_score     = float(get("depression_score", 2))
        sleep_hours          = float(get("sleep_hours", 7))
        physical_activity    = float(get("physical_activity", 3))
        social_support       = float(get("social_support", 5))
        screen_time          = float(get("screen_time", 4))
        internet_usage       = float(get("internet_usage", 4))
        financial_stress     = float(get("financial_stress", 3))
        family_expectation   = float(get("family_expectation", 5))

        # ── Encode Gender ──
        try:
            gender_enc = le_gender.transform([gender_raw])[0]
        except Exception:
            gender_enc = 0  # default Male

        # ── Build DataFrame ──
        data = pd.DataFrame([{
            "age":                  age,
            "gender":               gender_enc,
            "academic_year":        academic_year,
            "study_hours_per_day":  study_hours,
            "exam_pressure":        exam_pressure,
            "academic_performance": academic_performance,
            "stress_level":         stress_level,
            "anxiety_score":        anxiety_score,
            "depression_score":     depression_score,
            "sleep_hours":          sleep_hours,
            "physical_activity":    physical_activity,
            "social_support":       social_support,
            "screen_time":          screen_time,
            "internet_usage":       internet_usage,
            "financial_stress":     financial_stress,
            "family_expectation":   family_expectation,
        }])[feature_cols]

        # ── Scale & Predict ──
        data_scaled  = scaler.transform(data)
        pred_encoded = model.predict(data_scaled)[0]
        proba        = model.predict_proba(data_scaled)[0]
        risk         = le_target.inverse_transform([pred_encoded])[0]

        # ── Confidence ──
        confidence = round(float(max(proba)) * 100, 1)
        class_probs = {
            cls: round(float(p) * 100, 1)
            for cls, p in zip(le_target.classes_, proba)
        }

        # ── Save to DB ──
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO predictions (
                name, age, gender, academic_year, study_hours, exam_pressure,
                academic_performance, stress_level, anxiety_score, depression_score,
                sleep_hours, physical_activity, social_support, screen_time,
                internet_usage, financial_stress, family_expectation, risk_level, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            name, age, gender_raw, academic_year, study_hours, exam_pressure,
            academic_performance, stress_level, anxiety_score, depression_score,
            sleep_hours, physical_activity, social_support, screen_time,
            internet_usage, financial_stress, family_expectation, risk,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        conn.close()

        return jsonify({
            "name":        name,
            "risk":        risk,
            "confidence":  confidence,
            "class_probs": class_probs,
            "advice":      get_advice(risk)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/history", methods=["GET"])
def history():
    """Return last N predictions from DB."""
    try:
        limit = int(request.args.get("limit", 20))
        conn  = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/stats", methods=["GET"])
def stats():
    """Return aggregate stats from prediction history."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()

        total  = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        high   = c.execute("SELECT COUNT(*) FROM predictions WHERE risk_level='High'").fetchone()[0]
        medium = c.execute("SELECT COUNT(*) FROM predictions WHERE risk_level='Medium'").fetchone()[0]
        low    = c.execute("SELECT COUNT(*) FROM predictions WHERE risk_level='Low'").fetchone()[0]
        recent = c.execute(
            "SELECT name, risk_level, created_at FROM predictions ORDER BY id DESC LIMIT 5"
        ).fetchall()

        conn.close()
        return jsonify({
            "total_predictions": total,
            "high_risk":         high,
            "medium_risk":       medium,
            "low_risk":          low,
            "recent": [
                {"name": r[0], "risk": r[1], "time": r[2]} for r in recent
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/delete/<int:record_id>", methods=["DELETE"])
def delete_record(record_id):
    """Delete a prediction record by ID."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM predictions WHERE id=?", (record_id,))
        conn.commit()
        conn.close()
        return jsonify({"message": f"Record {record_id} deleted."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
