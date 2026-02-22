import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data_loader import load_data


def train_model():
    # Load dataset
    data = load_data("data/students.csv")

    if data is None:
        return None

    # Create average score
    data["average_score"] = (
        data["math score"] +
        data["reading score"] +
        data["writing score"]
    ) / 3

    # Create target column: Pass (1) or Fail (0)
    data["result"] = data["average_score"].apply(
        lambda x: 1 if x >= 40 else 0
    )

    # Features and target
    X = data[["math score", "reading score", "writing score"]]
    y = data["result"]

    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Check accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return model