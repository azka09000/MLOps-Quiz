from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Load Wine classification dataset
wine = load_wine()
X = wine.data
y = wine.target

# Use the same split as train.py
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load the trained model
model = joblib.load("models/model.pkl")

# Make predictions
predictions = model.predict(X_test)

# Evaluate the saved model
accuracy = accuracy_score(y_test, predictions)

print(f"Model: {type(model).__name__}")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))
