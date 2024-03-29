"""Seldon model class for Iris inference."""
import errno
import logging
import os
import pickle  # noqa - S403

from iris_inference_service.config import app_cfg

import numpy as np

import pandas as pd


class Iris:
    """Iris inference model class."""

    def __init__(self, model_name=app_cfg.model.name, model_file=app_cfg.model.file):
        """Seldon class init."""
        self.loaded = False
        self.model_name = model_name
        self.model_file = model_file

    def load(self):
        """Load machine learning model."""
        logging.info(f"Loading model from {self.model_file}")

        try:
            self.model = pickle.load(open(self.model_file, "rb"))  # noqa - S301
        except IOError:
            logging.exception(f"Unable to load the model file: {self.model_file}")
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), self.model_file
            )

        self.loaded = True

    def predict(self, X, features_names):
        """Prediction method for model class."""
        logging.debug(
            f"Performing prediction on X: {X}, features_names: {features_names}"
        )

        if not self.loaded:
            self.load()

        try:
            df = pd.DataFrame(data=X, columns=features_names)
        except ValueError:
            logging.exception("Unable to construct the dataframe.")
            raise ValueError

        pred = self.model.predict(df)

        logging.debug(
            f"Prediction result from X: {X}, features_names: {features_names}, pred: {pred}"
        )

        return pred

    def health_status(self):
        """Health endpoint validation for model class."""
        X_example = [[7.2, 3.6, 6.1, 2.5]]
        columns = ["sepalLength", "sepalWidth", "petalLength", "petalWidth"]

        response = self.predict(X_example, columns)

        if not type(response) == np.ndarray:
            raise AssertionError(
                "Prediction did not return the expected type of numpy.ndarray; "
                f"received {type(response)} instead."
            )

        if not len(response) == 1:
            raise AssertionError(
                "Prediction did not return a result of the expected length of 1; "
                f"received length of {len(response)} instead."
            )

        logging.debug("Health endpoint check successful.")

        return response
