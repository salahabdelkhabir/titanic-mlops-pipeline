import os

import pandas as pd
from sklearn.metrics import classification_report


def evaluate_and_save(pipeline, X_test, y_test, name: str, reports_path: str) -> None:
    """
    Generate a classification report for the given pipeline and save it
    as a CSV to reports_path/<name>_report.csv.
    """
    preds = pipeline.predict(X_test)
    report_dict = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(4)

    os.makedirs(reports_path, exist_ok=True)
    out_path = os.path.join(reports_path, f"{name}_report.csv")
    report_df.to_csv(out_path)

    print(f"  Report saved → {out_path}")
    print(report_df.to_string())
    print()