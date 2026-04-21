from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_models(cfg) -> dict:
    """Return a dictionary of model name -> estimator, built from Hydra config."""
    return {
        "log_reg": LogisticRegression(
            max_iter=cfg.pipeline.model.log_reg.max_iter,
            random_state=cfg.pipeline.model.log_reg.random_state,
        ),
        "rf": RandomForestClassifier(
            n_estimators=cfg.pipeline.model.rf.n_estimators,
            random_state=cfg.pipeline.model.rf.random_state,
        ),
    }