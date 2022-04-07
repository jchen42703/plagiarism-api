from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import KFold
from typing import List
import pandas as pd
from pandas.api.types import is_numeric_dtype
import os


class TrainPipeline(object):
    def __init__(
        self,
        export_dir: str,
        train_embeds_path: str,
        train_labels_path: str,
        num_folds: int = 5,
        seed: int = 1881,
        num_epochs: int = 20000,
        early_stopping_rounds: int = 3000,
        cpu_only: bool = True,
        cat_model_params: dict = {},
        fit_params: dict = {},
    ) -> None:
        """
        Args:
            train_embeds_path: Should be the path to 'cat_train_values.csv'
                The difference between the cat values and the original values is that
                it contains the embeds but categorical features ARE NOT ONE HOT ENCODED
                because catboost prefers non one-hot encoded features.
                - Note: superstructure and has_secondary_use related columns are still
                left as one hot encoded because the features may not be entirely
                mutually exclusive (multi-label instead of multi-class)
        """
        super().__init__()
        self.export_dir = export_dir
        if not os.path.isdir(self.export_dir):
            print("Creating export dir: ", self.export_dir)
            os.makedirs(self.export_dir)

        self.X = pd.read_csv(train_embeds_path)
        self.Y = pd.read_csv(train_labels_path).drop(["building_id"], axis=1)

        assert len(self.X) == len(
            self.Y
        ), "There should be the same number of rows for the input and the labels"

        self.num_folds = num_folds
        self.seed = seed
        self.num_epochs = num_epochs
        self.early_stopping_rounds = early_stopping_rounds
        self.cpu_only = cpu_only

        # additional params
        self.cat_model_params = cat_model_params
        self.fit_params = fit_params

    def create_split(self, train_index: List[int], test_index: List[int]):
        return (
            self.X.iloc[train_index],
            self.X.iloc[test_index],
            self.Y.iloc[train_index],
            self.Y.iloc[test_index],
        )

    def get_categorical_idx(self) -> List[int]:
        categorical_indicies = get_categorical_indicies(self.X)
        return categorical_indicies

    def train_single_fold(
        self,
        fold: int,
        train_index: List[int],
        test_index: List[int],
        categorical_indices: List[int],
    ):
        """Trains a single fold for catboost."""

        X_train, X_test, Y_train, Y_test = self.create_split(train_index, test_index)
        train_dataset = Pool(X_train, Y_train, cat_features=categorical_indices)
        test_dataset = Pool(X_test, Y_test, cat_features=categorical_indices)

        params = {
            "iterations": self.num_epochs,
            "loss_function": "MultiClass",
            "task_type": "CPU" if self.cpu_only else "GPU",
            "random_seed": self.seed,
            "early_stopping_rounds": self.early_stopping_rounds,
            "train_dir": self.export_dir,
            **self.cat_model_params,
        }

        clf = CatBoostClassifier(**params)

        clf.fit(train_dataset, eval_set=test_dataset, **self.fit_params)

        # Gets the performance of the last tree
        metric_key = "TotalF1:average=Micro"
        f1 = clf.eval_metrics(test_dataset, [metric_key],)[
            metric_key
        ][-1]
        print("CatBoost model is fitted: " + str(clf.is_fitted()))
        print("CatBoost model parameters: \n", clf.get_params())
        print("F1: ", f1)

        save_path = os.path.join(self.export_dir, f"catboost_fold{fold}.cbm")
        clf.save_model(save_path)
        print(f"Saved model: {save_path}")

    def train_kfold(self):
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        categorical_indices = self.get_categorical_idx()
        convert_cats(self.X)
        for idx, (train_index, test_index) in enumerate(kf.split(self.X)):
            print(f"===== Training for fold {idx+1} =====")
            self.train_single_fold(
                idx + 1, train_index, test_index, categorical_indices
            )


def get_categorical_indicies(X):
    """Get the indices of all columns that are categorical (not numerical)"""
    cats = []
    for col in X.columns:
        if is_numeric_dtype(X[col]):
            pass
        else:
            cats.append(col)

    cat_indicies = []
    for col in cats:
        cat_indicies.append(X.columns.get_loc(col))

    return cat_indicies


def convert_cats(df: pd.DataFrame):
    """Converts categorical columns to type 'category'"""
    cats = []
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            pass
        else:
            cats.append(col)

    for col in cats:
        df[col] = df[col].astype("category")
    return df
