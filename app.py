from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os

from sklearn.metrics.pairwise import rbf_kernel

def quantum_inspired_kernel(X, Y=None, gamma=0.5):
    return rbf_kernel(X, Y, gamma=gamma)

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load models
svm = joblib.load("models/svm.pkl")
rf = joblib.load("models/rf.pkl")
qsvm = joblib.load("models/qsvm.pkl")
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")

label_names = {
    0: "Normal",
    1: "Myocardial Infarction",
    2: "ST/T Change",
    3: "Conduction Disturbance",
    4: "Hypertrophy"
}
disease_descriptions = {
    "Normal": "The ECG signal indicates normal heart activity with no significant abnormalities detected.",
    
    "Myocardial Infarction":
        "Indicates possible heart muscle damage due to reduced blood supply. Immediate medical attention is advised.",
    
    "ST/T Change":
        "Represents abnormalities in the ST segment or T wave, which may suggest ischemia or electrolyte imbalance.",
    
    "Conduction Disturbance":
        "Shows irregular electrical conduction in the heart, potentially affecting heartbeat rhythm.",
    
    "Hypertrophy":
        "Suggests thickening of heart muscle walls, often associated with high blood pressure or cardiac stress."
}

def extract_features_12lead(ecg):
    features = []
    for i in range(ecg.shape[1]):
        lead = ecg[:, i]
        features.extend([
            np.mean(lead),
            np.std(lead),
            np.min(lead),
            np.max(lead),
            np.var(lead)
        ])
    return np.array(features).reshape(1, -1)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    description = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            df = pd.read_csv(file)

            # Feature extraction (same as training)
            features = []
            for col in df.columns:
                lead = df[col].values
                features.extend([
                    lead.mean(),
                    lead.std(),
                    lead.min(),
                    lead.max(),
                    lead.var()
                ])

            X = np.array(features).reshape(1, -1)

            # Preprocessing
            X_scaled = scaler.transform(X)
            X_pca = pca.transform(X_scaled)

            # Predictions
            pred_svm = svm.predict(X_scaled)
            pred_rf = rf.predict(X_scaled)
            pred_qsvm = qsvm.predict(X_pca)

            # Majority voting
            final_pred = int(
                np.bincount(
                    np.array([pred_svm[0], pred_rf[0], pred_qsvm[0]])
                ).argmax()
            )

            prediction = label_names[final_pred]
            description = disease_descriptions[prediction]

    return render_template(
        "index.html",
        prediction=prediction,
        description=description
    )

if __name__ == "__main__":
    app.run(debug=True)
