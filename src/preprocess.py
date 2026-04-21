from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_FEATURES = ["Age", "Fare"]
CAT_FEATURES = ["Sex", "Embarked", "Pclass"]


def build_preprocessor() -> ColumnTransformer:
    """Build and return the feature preprocessing pipeline."""
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", num_pipeline, NUM_FEATURES),
            ("cat", cat_pipeline, CAT_FEATURES),
        ]
    )

    return preprocessor