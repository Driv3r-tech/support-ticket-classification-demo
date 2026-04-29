import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def build_preprocessor(columns: list[str]) -> ColumnTransformer:
    transformers = []

    if "message" in columns:
        transformers.append(("message_tfidf", TfidfVectorizer(), "message"))

    categorical_columns = [column for column in columns if column != "message"]

    if categorical_columns:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns))

    return ColumnTransformer(transformers=transformers)


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", None)

    df = pd.read_json("data/clean_requests_100.json")
    y = df["category"]

    feature_sets = {
        "message_only": ["message"],
        "message_priority": ["message", "priority"],
        "message_user_priority": ["message", "user", "priority"],
        "all_features": ["message", "user", "status", "priority"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for set_name, columns in feature_sets.items():
        X = df[columns]
        preprocessor = build_preprocessor(columns)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(random_state=42))
            ]
        )

        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro")

        results.append({
            "feature_set": set_name,
            "columns": columns,
            "macro_f1_mean": scores.mean(),
            "macro_f1_std": scores.std(),
            "scores": scores
        })

    results_df = pd.DataFrame(results)
    results_df["macro_f1_mean"] = results_df["macro_f1_mean"].round(3)
    results_df["macro_f1_std"] = results_df["macro_f1_std"].round(3)
    results_df = results_df.sort_values(by="macro_f1_mean", ascending=False)

    print(results_df)
    results_df.to_csv("results/leakage_check.csv", index=False)


if __name__ == "__main__":
    main()
