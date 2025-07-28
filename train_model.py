import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("training_data.csv")

X = df.drop("label", axis=1)
y = df["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

joblib.dump(model, "app/model.pkl")
joblib.dump(le, "app/label_encoder.pkl")

print("Model & label encoder saved.")
