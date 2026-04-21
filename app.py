
from flask import Flask, request, jsonify, render_template_string
import joblib, json, numpy as np, pandas as pd

app = Flask(__name__)

model  = joblib.load("house_model.pkl")
scaler = joblib.load("house_scaler.pkl")
with open("model_meta.json") as f:
    meta = json.load(f)

FEATURE_COLS = meta["feature_cols"]

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>House Price Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0f1117; --card: #1a1d27; --accent: #6C63FF;
    --accent2: #00D9A3; --text: #e8eaf6; --muted: #7c82a0;
    --border: #2a2d3e; --danger: #ff6b6b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif;
         min-height: 100vh; padding: 2rem 1rem; }
  .container { max-width: 780px; margin: auto; }
  header { text-align: center; margin-bottom: 2.5rem; }
  header h1 { font-family: 'DM Serif Display', serif; font-size: 2.6rem;
               background: linear-gradient(135deg, var(--accent), var(--accent2));
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  header p  { color: var(--muted); margin-top: .5rem; font-size: .95rem; }
  .stats-bar { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
  .stat { flex: 1; min-width: 140px; background: var(--card); border: 1px solid var(--border);
           border-radius: 12px; padding: 1rem; text-align: center; }
  .stat .val { font-size: 1.4rem; font-weight: 600; color: var(--accent2); }
  .stat .lbl { font-size: .78rem; color: var(--muted); margin-top: .25rem; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 2rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; }
  label { display: block; font-size: .85rem; color: var(--muted); margin-bottom: .4rem; font-weight: 500; }
  input, select {
    width: 100%; padding: .65rem .9rem; background: #12141e;
    border: 1px solid var(--border); border-radius: 8px;
    color: var(--text); font-size: .95rem; font-family: inherit;
    transition: border .2s;
  }
  input:focus, select:focus { outline: none; border-color: var(--accent); }
  select option { background: #1a1d27; }
  .btn { width: 100%; margin-top: 1.8rem; padding: .9rem; border: none; border-radius: 10px;
          background: linear-gradient(135deg, var(--accent), var(--accent2));
          color: #fff; font-size: 1.05rem; font-weight: 600; cursor: pointer;
          font-family: inherit; letter-spacing: .02em; transition: opacity .2s; }
  .btn:hover { opacity: .88; }
  .result-box { margin-top: 2rem; display: none; background: linear-gradient(135deg,#1d1a3a,#0d2a24);
                border: 1px solid var(--accent); border-radius: 14px; padding: 1.8rem; text-align: center; }
  .result-box .price { font-family: 'DM Serif Display', serif; font-size: 2.8rem;
                        background: linear-gradient(90deg,var(--accent),var(--accent2));
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .result-box .sub  { color: var(--muted); font-size: .9rem; margin-top: .5rem; }
  .error-box { margin-top: 1.5rem; background: #2a1515; border: 1px solid var(--danger);
                border-radius: 10px; padding: 1rem; color: var(--danger); display: none; }
  .section-title { font-size: .8rem; text-transform: uppercase; letter-spacing: .1em;
                    color: var(--muted); margin-bottom: 1.2rem; border-bottom: 1px solid var(--border);
                    padding-bottom: .5rem; }
  @media(max-width:540px){ .grid{ grid-template-columns:1fr; } }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>🏠 House Price Predictor</h1>
    <p>ML-powered estimates using Linear Regression · Ridge · Lasso</p>
  </header>

  <div class="stats-bar">
    <div class="stat"><div class="val">{{ meta.best_model.split(' ')[0] }}</div><div class="lbl">Best Algorithm</div></div>
    <div class="stat"><div class="val">{{ meta.r2 }}</div><div class="lbl">R² Score</div></div>
    <div class="stat"><div class="val">${{ "{:,.0f}".format(meta.rmse) }}</div><div class="lbl">RMSE</div></div>
    <div class="stat"><div class="val">${{ "{:,.0f}".format(meta.mae) }}</div><div class="lbl">MAE</div></div>
  </div>

  <div class="card">
    <div class="section-title">Property Details</div>
    <div class="grid">
      <div>
        <label>Location</label>
        <select id="location">
          <option>Suburbs</option><option>Downtown</option>
          <option>Midtown</option><option>Rural</option><option>Waterfront</option>
        </select>
      </div>
      <div>
        <label>Condition</label>
        <select id="condition">
          <option>Good</option><option>Excellent</option>
          <option>Fair</option><option>Poor</option>
        </select>
      </div>
      <div><label>Square Footage</label><input type="number" id="sqft" value="1800" min="300" max="10000"></div>
      <div><label>Bedrooms</label><input type="number" id="bedrooms" value="3" min="1" max="10"></div>
      <div><label>Bathrooms</label><input type="number" id="bathrooms" value="2" min="1" max="8"></div>
      <div><label>Property Age (years)</label><input type="number" id="age" value="10" min="0" max="100"></div>
      <div>
        <label>Garage Spaces</label>
        <select id="garage"><option value="0">None</option><option value="1">1 Car</option><option value="2">2 Cars</option></select>
      </div>
      <div>
        <label>Floors</label>
        <select id="floors"><option value="1">1</option><option value="2">2</option><option value="3">3</option></select>
      </div>
      <div>
        <label>Swimming Pool</label>
        <select id="pool"><option value="0">No</option><option value="1">Yes</option></select>
      </div>
    </div>
    <button class="btn" onclick="predict()">⚡ Predict Price</button>
    <div class="result-box" id="resultBox">
      <div class="price" id="priceDisplay"></div>
      <div class="sub" id="priceSub"></div>
    </div>
    <div class="error-box" id="errorBox"></div>
  </div>
</div>

<script>
async function predict() {
  const data = {
    location:  document.getElementById('location').value,
    condition: document.getElementById('condition').value,
    sqft:      +document.getElementById('sqft').value,
    bedrooms:  +document.getElementById('bedrooms').value,
    bathrooms: +document.getElementById('bathrooms').value,
    age:       +document.getElementById('age').value,
    garage:    +document.getElementById('garage').value,
    floors:    +document.getElementById('floors').value,
    pool:      +document.getElementById('pool').value,
  };
  document.getElementById('resultBox').style.display = 'none';
  document.getElementById('errorBox').style.display  = 'none';
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data)
    });
    const json = await res.json();
    if (json.error) throw new Error(json.error);
    const price = json.predicted_price;
    document.getElementById('priceDisplay').textContent = '$' + price.toLocaleString();
    document.getElementById('priceSub').textContent = 'Estimated market value · ' + data.sqft + ' sq ft · ' + data.location;
    document.getElementById('resultBox').style.display = 'block';
  } catch(e) {
    document.getElementById('errorBox').textContent = 'Error: ' + e.message;
    document.getElementById('errorBox').style.display = 'block';
  }
}
</script>
</body></html>'''  

@app.route("/")
def index():
    return render_template_string(HTML, meta=meta)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.get_json()
        sqft      = float(d["sqft"])
        bedrooms  = float(d["bedrooms"])
        bathrooms = float(d["bathrooms"])
        age       = float(d["age"])
        garage    = float(d["garage"])
        pool      = float(d["pool"])
        floors    = float(d["floors"])
        location  = d["location"]
        condition = d["condition"]

        row = {
            "SqFt":sqft, "Bedrooms":bedrooms, "Bathrooms":bathrooms,
            "Age":age,   "Garage":garage,     "Pool":pool, "Floors":floors,
            "IsNew":     int(age<=5),
            "HasAmenities": int(pool==1 or garage>=1),
            "RoomRatio": bedrooms/(bathrooms+1),
            "TotalRooms":bedrooms+bathrooms,
            "SqFt_log":  np.log1p(sqft),
            "Age_sq":    age**2,
        }
        for loc in ["Downtown","Midtown","Rural","Suburbs","Waterfront"]:
            key = f"Location_{loc}"
            if key in FEATURE_COLS:
                row[key] = 1 if location==loc else 0
        for cond in ["Excellent","Fair","Good","Poor"]:
            key = f"Condition_{cond}"
            if key in FEATURE_COLS:
                row[key] = 1 if condition==cond else 0

        X = pd.DataFrame([row])
        for c in FEATURE_COLS:
            if c not in X.columns:
                X[c] = 0
        X = X[FEATURE_COLS]
        X_s = scaler.transform(X)
        price = float(model.predict(X_s)[0])
        price = max(50000, round(price, -2))
        return jsonify({"predicted_price": price})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(port=5000, debug=False, use_reloader=False)
