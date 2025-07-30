

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # 👈 VERY IMPORTANT

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    url = data.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    features = vectorizer.transform([url])
    prediction = model.predict(features)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]
    return jsonify({"result": predicted_label})

if __name__ == "__main__":
    app.run(debug=True)
