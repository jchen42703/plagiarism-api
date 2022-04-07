import lightgbm as lgb
from catboost import CatBoostClassifier
import numpy as np


class ModelWrapper(object):
    """Wrapper for LightGBM/Catboost models for prediction."""

    def __init__(self, model_type: str, model_path: str = None) -> None:
        # supported_models = ["lightgbm", "catboost"]
        # add this back when catboost is supported
        supported_models = ["lightgbm", "catboost"]
        assert (
            model_type.lower() in supported_models
        ), f"{model_type} is not one of: {supported_models}"

        self.model_type = model_type.lower()

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, path: str):
        if self.model_type == "lightgbm":
            self.model = lgb.Booster(model_file=path)
        elif self.model_type == "catboost":
            self.model = CatBoostClassifier()
            self.model.load_model(path)
        else:
            raise NotImplementedError
        return self.model

    def num_features(self):
        """Number of input features that the model expects"""
        if self.model_type == "lightgbm":
            return self.model.num_feature()
        elif self.model_type == "catboost":
            return len(self.model.get_feature_importance())
        else:
            raise NotImplementedError

    def predict(self, x: np.ndarray, do_argmax: bool = True) -> np.ndarray:
        """Predicts for an array, x

        Args:
            x: Should be an array of shape (batch_size, 81) for the models
                using the data with embeds

        Returns:
            An array with the damage grade predictions
                Ex: Shape (batch_size,)
                    array([2, 2])
            If do_argmax == False, then probabilities will be returned instead:
                i.e. shape (batch_size, 3)
                array([[0.30518331, 0.36225706, 0.33255963],
                       [0.30518331, 0.36225706, 0.33255963]])
        """
        if self.model_type == "lightgbm":
            y_pred = self.model.predict(x, num_threads=1)
        elif self.model_type == "catboost":
            y_pred = self.model.predict_proba(x, thread_count=1)

        if do_argmax:
            return y_pred.argmax(axis=1) + 1
        else:
            return y_pred
