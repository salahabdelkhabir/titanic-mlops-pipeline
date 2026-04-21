from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_models():
    return {
        "log_reg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(),
    }