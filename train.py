"""Train a simple classifier on the Iris dataset and save it as model.joblib."""

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Load data
iris = load_iris()
X, y = iris.data, iris.target

# 2. Train / test split (stratified so all 3 classes are balanced in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 5. Save the model together with the class names (the artifact the API will load)
joblib.dump(
    {"model": model, "target_names": list(iris.target_names)},
    "model.joblib",
)
print("Model saved to model.joblib")
