from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np
import joblib
from typing import List


class Encoder(object):
    """For managing the encoder model and replacing geo_level embeddings"""

    def __init__(self, weights_path: str, one_hot_cols: str, one_hot_geo_path: str):
        self.model = None
        self.one_hot_cols = one_hot_cols
        self.geo_one_hot: OneHotEncoder = joblib.load(one_hot_geo_path)

        if weights_path is not None:
            self.create_model(weights_path)

    def __autoencoder(
        self,
        geo1_classes: int = 31,
        geo2_classes: int = 1418,
        geo3_classes: int = 11861,
    ):
        """
        Args:
            geo1_classes: corresponds to the number of unique ids in the
                geo_level_1_id column.
            geo2_classes: ...
            geo3_classes: ...
        """
        inp = Input((geo3_classes,))
        i1 = Dense(16, name="intermediate")(inp)
        x2 = Dense(geo2_classes, activation="sigmoid")(i1)
        x1 = Dense(geo1_classes, activation="sigmoid")(i1)

        model = Model(inp, [x2, x1])
        model.compile(loss="binary_crossentropy", optimizer="adam")
        return model

    def create_model(self, weights_path: str):
        """Creates the encoder and loads weights if provided"""
        self.model = self.__autoencoder()
        if weights_path != None:
            self.model.load_weights(weights_path)
        return self.model

    def get_embedding_layer(self):
        get_int_layer_output = K.function(
            [self.model.layers[0].input], [self.model.layers[1].output]
        )
        return get_int_layer_output

    def replace_with_new_embeds(
        self, df: pd.DataFrame, batch_size: int
    ) -> pd.DataFrame:
        """Edits the dataframe in_place"""
        # Extract GEO-Embeds for all train data points.

        # One hot encode with a previous fitted transform (used for training)
        geo3 = self.geo_one_hot.transform(
            np.array(df["geo_level_3_id"])[None]
        ).toarray()
        # Get embeds from encoder
        get_int_layer_output = self.get_embedding_layer()

        out = []
        for i in range(0, len(df), batch_size):
            layer_output = get_int_layer_output(geo3[i : i + batch_size])[0]
            out.append(layer_output)

        out = np.array(out).squeeze(axis=0)

        new_df = one_hot_encode(df, self.one_hot_cols)
        params = {f"geo_feat{idx+1}": out[:, idx] for idx in range(16)}
        augmented_df = new_df.assign(**params)
        return augmented_df


def drop_useless_cols(df: pd.DataFrame):
    """Specific function to drop columns not needed for one hot encoding"""
    if "building_id" in df.columns:
        df = df.drop(["building_id"], axis=1)

    if "Unnamed: 0" in df.columns:
        df = df.drop(["Unnamed: 0"], axis=1)
    df = df.drop(["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"], axis=1)
    return df


def one_hot_encode(df: pd.DataFrame, one_hot_cols: List[str]):
    """One-hot encodes during the prediction pipeline. Normally pd.get_dummies does not
    know about all of the existing features. But, by specifying existing columns, it's
    okay.
    """
    new_concat = drop_useless_cols(df)
    return pd.get_dummies(new_concat).reindex(columns=one_hot_cols, fill_value=0)
