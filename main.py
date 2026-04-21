import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def main():
    # 1. Load data
    df = pd.read_csv("data/train.csv")

    # 2. Basic preprocessing
    df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]]
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    # 3. Split
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Models (2 models ✔️)
    models = {
        "log_reg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier()
    }

    # 5. Train + evaluate + save
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        print(f"{name} accuracy: {acc}")

        joblib.dump(model, f"models/{name}.joblib")


if __name__ == "__main__":
    main()