from flask import Flask ,request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

# Load trained Random Forest model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR , "data","crop_model.pkl")
model = None
model_error = None
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model_error = str(e)

@app.route("/")
def home():
    return render_template("index.html" , prediction=None,error=model_error)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = [[
        float(data["N"]),
        float(data["P"]),
        float(data["K"]),
        float(data["temperature"]),
        float(data["humidity"]),
        float(data["ph"]),
        float(data["rainfall"])
    ]]

    prediction = model.predict(features)[0]

    return jsonify({
        "crop": prediction
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT",10000))
    app.run(host="0.0.0.0",port=port)                          
