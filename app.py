"""
Flask App - Placement Package Predictor
Loads placement_lr.pkl (Simple Linear Regression model) and exposes:
  GET  /          -> Serves the prediction form
  POST /predict   -> Returns the predicted package (in Lacs) given a CGPA input
"""

import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the trained model
model = joblib.load("placement_lr.pkl")

# ---------------------------------------------------------------------------
# HTML template (served at GET /)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Placement Package Predictor</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
      font-family: 'Inter', sans-serif;
      color: #fff;
    }

    .card {
      background: rgba(255, 255, 255, 0.07);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 24px;
      padding: 48px 40px;
      width: 100%;
      max-width: 480px;
      box-shadow: 0 24px 64px rgba(0, 0, 0, 0.4);
      animation: fadeIn 0.6s ease;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(24px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .badge {
      display: inline-block;
      background: linear-gradient(90deg, #7f53ac, #647dee);
      border-radius: 50px;
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      padding: 4px 14px;
      margin-bottom: 20px;
      text-transform: uppercase;
    }

    h1 {
      font-size: 1.9rem;
      font-weight: 700;
      line-height: 1.25;
      margin-bottom: 8px;
    }

    .subtitle {
      font-size: 0.9rem;
      color: rgba(255,255,255,0.55);
      margin-bottom: 36px;
    }

    label {
      display: block;
      font-size: 0.82rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      color: rgba(255,255,255,0.7);
      margin-bottom: 8px;
      text-transform: uppercase;
    }

    input[type="number"] {
      width: 100%;
      padding: 14px 18px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.2);
      background: rgba(255,255,255,0.08);
      color: #fff;
      font-size: 1.05rem;
      font-family: 'Inter', sans-serif;
      outline: none;
      transition: border-color 0.25s, box-shadow 0.25s;
      margin-bottom: 24px;
    }

    input[type="number"]:focus {
      border-color: #7f53ac;
      box-shadow: 0 0 0 3px rgba(127, 83, 172, 0.3);
    }

    input[type="number"]::placeholder { color: rgba(255,255,255,0.3); }

    button {
      width: 100%;
      padding: 15px;
      border: none;
      border-radius: 12px;
      background: linear-gradient(90deg, #7f53ac, #647dee);
      color: #fff;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.2s, transform 0.15s;
      letter-spacing: 0.03em;
    }

    button:hover  { opacity: 0.88; transform: translateY(-1px); }
    button:active { transform: translateY(0); }

    .result {
      margin-top: 28px;
      padding: 20px 24px;
      border-radius: 14px;
      background: linear-gradient(135deg, rgba(127,83,172,0.25), rgba(100,125,238,0.25));
      border: 1px solid rgba(127,83,172,0.4);
      text-align: center;
      animation: fadeIn 0.4s ease;
    }

    .result p { font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-bottom: 6px; }

    .result .value {
      font-size: 2.4rem;
      font-weight: 700;
      background: linear-gradient(90deg, #c471ed, #a2d2ff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .error {
      margin-top: 20px;
      padding: 14px 18px;
      border-radius: 10px;
      background: rgba(255, 80, 80, 0.15);
      border: 1px solid rgba(255,80,80,0.4);
      font-size: 0.88rem;
      color: #ff8080;
    }
  </style>
</head>
<body>
  <div class="card">
    <span class="badge">ML Model</span>
    <h1>Placement Package Predictor</h1>
    <p class="subtitle">Enter your CGPA to estimate your placement package using a trained Linear Regression model.</p>

    <form method="POST" action="/predict" id="predict-form">
      <label for="cgpa">CGPA (4.0 – 10.0)</label>
      <input
        type="number"
        id="cgpa"
        name="cgpa"
        step="0.01"
        min="4"
        max="10"
        placeholder="e.g. 7.5"
        value="{{ cgpa or '' }}"
        required
      />
      <button type="submit">Predict Package</button>
    </form>

    {% if prediction is not none %}
    <div class="result">
      <p>Estimated Package</p>
      <div class="value">₹ {{ prediction }} Lacs</div>
    </div>
    {% endif %}

    {% if error %}
    <div class="error">⚠ {{ error }}</div>
    {% endif %}
  </div>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, prediction=None, error=None, cgpa=None)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts form data (cgpa) and returns a rendered page with the prediction,
    OR returns JSON when the request prefers JSON (for API clients).
    """
    try:
        cgpa = float(request.form.get("cgpa") or request.json.get("cgpa"))
        if not (0 <= cgpa <= 10):
            raise ValueError("CGPA must be between 0 and 10.")

        prediction = model.predict(np.array([[cgpa]]))[0]
        prediction_rounded = round(float(prediction), 2)

        # API client (e.g. Postman / curl)
        if request.is_json:
            return jsonify({"cgpa": cgpa, "predicted_package_lacs": prediction_rounded})

        # Browser form submission
        return render_template_string(
            HTML_TEMPLATE,
            prediction=prediction_rounded,
            error=None,
            cgpa=cgpa,
        )

    except (TypeError, ValueError, AttributeError) as e:
        error_msg = str(e) if str(e) else "Invalid input. Please enter a numeric CGPA."
        if request.is_json:
            return jsonify({"error": error_msg}), 400
        return render_template_string(HTML_TEMPLATE, prediction=None, error=error_msg, cgpa=None)


if __name__ == "__main__":
    app.run(debug=True)
