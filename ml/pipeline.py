import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def main() -> None:
    df = pd.read_json("data/clean_requests_100.json")

    X = df[["message", "user", "status", "priority"]]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("message_tfidf", TfidfVectorizer(), "message"),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["user", "status", "priority"])
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42))
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nPipeline accuracy:")
    print(accuracy_score(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
