import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def train_and_save(
    model,
    preprocessor,
    X_train,
    y_train,
    X_test,
    y_test,
    name: str,
    model_path: str = "models",
) -> Pipeline:
    """
    Build a full sklearn Pipeline, fit it, print accuracy,
    and persist it to disk. Returns the fitted pipeline.
    """
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[{name}] accuracy: {acc:.4f}")

    os.makedirs(model_path, exist_ok=True)
    save_path = os.path.join(model_path, f"{name}.joblib")
    joblib.dump(pipeline, save_path)
    print(f"  Model saved → {save_path}")

    return pipeline