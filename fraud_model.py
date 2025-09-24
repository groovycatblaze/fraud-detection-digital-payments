import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Generate synthetic dataset (mimicking payment transactions)
def generate_data():
    import numpy as np
    n = 5000
    data = {
        "amount": np.random.uniform(1, 5000, n),
        "time": np.random.randint(0, 24, n),
        "location_id": np.random.randint(1, 100, n),
        "device_id": np.random.randint(1, 500, n),
        "is_foreign": np.random.randint(0, 2, n),
        "is_high_risk_country": np.random.randint(0, 2, n),
        "fraud": np.random.randint(0, 2, n)
    }
    return pd.DataFrame(data)

df = generate_data()
X = df.drop("fraud", axis=1)
y = df["fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

# Save model
joblib.dump(model, "fraud_model.pkl")
