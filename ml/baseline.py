import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report


def main() -> None:
    df = pd.read_json("data/clean_requests_100.json")

    print("Первые строки датасета:")
    print(df.head())

    print("\nРаспределение категорий:")
    print(df["category"].value_counts())

    X = df["message"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    y_pred = baseline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nBaseline accuracy:")
    print(accuracy)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
