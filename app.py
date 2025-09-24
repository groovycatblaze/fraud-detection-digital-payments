```python
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import shap

app = Flask(__name__)

# Load model
model = joblib.load("fraud_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        
        # SHAP explainability
        explainer = shap.Explainer(model, df)
        shap_values = explainer(df)
        explanation = shap_values.values.tolist()

        return jsonify({
            "prediction": int(prediction),
            "explanation": explanation
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
