import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


def main() -> None:
    df = pd.read_json("data/clean_requests_100.json")

    X = df[["message", "user", "status", "priority"]]
    y = df["category"]

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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for model_name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier)
            ]
        )

        accuracy_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        macro_f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro")

        results.append({
            "model": model_name,
            "accuracy_mean": accuracy_scores.mean(),
            "accuracy_std": accuracy_scores.std(),
            "macro_f1_mean": macro_f1_scores.mean(),
            "macro_f1_std": macro_f1_scores.std(),
            "accuracy_scores": accuracy_scores
        })

    results_df = pd.DataFrame(results)
    for col in ["accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"]:
        results_df[col] = results_df[col].round(3)

    results_df = results_df.sort_values(by="macro_f1_mean", ascending=False)
    print(results_df[["model", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std", "accuracy_scores"]])
    results_df.to_csv("results/cross_validation.csv", index=False)


if __name__ == "__main__":
    main()
