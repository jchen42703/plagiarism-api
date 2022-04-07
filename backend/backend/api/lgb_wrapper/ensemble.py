from typing import List
import lightgbm as lgb
import numpy as np
import pandas as pd
import os


class LightGBMEnsemble(object):
    """Wrapper for ensembling LightGBM models."""

    def __init__(self, model_paths: List[str], export_dir: str):
        self.model_paths = model_paths
        self.models: List[lgb.Booster] = []
        self.export_dir = export_dir

    def load_models(self):
        for path in self.model_paths:
            self.models.append(lgb.Booster(model_file=path))

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """Assumes that df already contains the embeds. Processes the df and
        converts it to a numpy array to be used in self.predict.
        """
        if "building_id" in df.columns:
            df = df.drop(["building_id"], axis=1)

        if "Unnamed: 0" in df.columns:
            df = df.drop(["Unnamed: 0"], axis=1)

        return np.array(df)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Actually calls the ensemble for argmaxes for damage grades 1, 2, 3"""
        y_pred = ensemble(self.models, x)
        return y_pred.argmax(axis=1) + 1

    def create_submission(self, sub_df: pd.DataFrame, test_df: pd.DataFrame):
        y_pred = self.predict(self.preprocess(test_df))
        sub_df["damage_grade"] = y_pred
        export_path = os.path.join(self.export_dir, "submission.csv")
        sub_df.to_csv(export_path, index=False)


def threshold_arr(array):
    # Get major confidence-scored predicted value.
    new_arr = []
    for ix, val in enumerate(array):
        loc = np.array(val).argmax(axis=0)
        k = list(np.zeros((len(val))))
        k[loc] = 1
        new_arr.append(k)

    return np.array(new_arr)


def ensemble(models, x):
    # Ensemble K-Fold CV models with adding all confidence score by class.
    y_preds = []

    for model in models:
        y_pred = model.predict(x)
        y_preds.append(y_pred)

    init_y_pred = y_preds[0]
    for ypred in y_preds[1:]:
        init_y_pred += ypred

    y_pred = threshold_arr(init_y_pred)

    return y_pred
