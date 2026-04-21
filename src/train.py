from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib


def train_and_save(
    model, preprocessor, X_train, y_train, X_test, y_test, name
):
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"{name} accuracy: {acc}")

    joblib.dump(pipeline, f"models/{name}.joblib")