from sklearn.model_selection import train_test_split

from src.data import load_data, ensure_dirs
from src.evaluate import evaluate_and_save
from src.models import get_models
from src.preprocess import build_preprocessor
from src.train import train_and_save


def main() -> None:
    ensure_dirs("models", "reports")

    df = load_data("data/train.csv")

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = get_models()

    for name, model in models.items():
        print(f"\n── Training: {name} ──")
        preprocessor = build_preprocessor()
        pipeline = train_and_save(
            model,
            preprocessor,
            X_train,
            y_train,
            X_test,
            y_test,
            name,
            model_path="models",
        )
        evaluate_and_save(pipeline, X_test, y_test, name, reports_path="reports")


if __name__ == "__main__":
    main()