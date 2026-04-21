import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from src.data import ensure_dirs, load_data
from src.evaluate import evaluate_and_save
from src.models import get_models
from src.preprocess import build_preprocessor
from src.train import train_and_save


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    ensure_dirs(
        cfg.pipeline.model.model_path,
        cfg.pipeline.evaluate.reports_path,
    )

    df = load_data(cfg.pipeline.data.raw_data_path)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.pipeline.data.test_size,
        random_state=cfg.pipeline.data.random_state,
    )

    models = get_models(cfg)

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
            model_path=cfg.pipeline.model.model_path,
        )
        evaluate_and_save(
            pipeline,
            X_test,
            y_test,
            name,
            reports_path=cfg.pipeline.evaluate.reports_path,
        )


if __name__ == "__main__":
    main()