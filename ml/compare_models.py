import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, classification_report


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

    models = {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "LinearSVC": LinearSVC(),
        "MultinomialNB": MultinomialNB(),
        "RandomForestClassifier": RandomForestClassifier(random_state=42)
    }

    results = []

    for model_name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier)
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        results.append({
            "model": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "weighted_f1": f1_score(y_test, y_pred, average="weighted")
        })

        print("\n" + "=" * 60)
        print(model_name)
        print("=" * 60)
        print(classification_report(y_test, y_pred, zero_division=0))

    results_df = pd.DataFrame(results).sort_values(by="weighted_f1", ascending=False)
    print("\nСравнение моделей:")
    print(results_df)
    results_df.to_csv("results/model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
