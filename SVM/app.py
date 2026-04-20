from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('svm_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return "API SVM films active"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    userId = data['userId']
    movieId = data['movieId']

    X = np.array([[userId, movieId]])
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]

    result = "Like" if pred == 1 else "Dislike"

    return jsonify({
        "prediction": int(pred),
        "result": result
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=100000)