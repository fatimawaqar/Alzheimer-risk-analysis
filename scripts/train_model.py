import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib
import os

# Load dataset
df = pd.read_csv("dataset/alzheimer_data.csv")

# Encode target label
le = LabelEncoder()
df["risk_level"] = le.fit_transform(df["risk_level"])
# Low=1, Medium=2, High=0 (internally)

# Features & target
X = df.drop("risk_level", axis=1)
y = df["risk_level"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
print("Model trained successfully!")
print("Accuracy:", round(acc, 2))
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/alzheimer_model.pkl")

print("Model saved in model/alzheimer_model.pkl")
