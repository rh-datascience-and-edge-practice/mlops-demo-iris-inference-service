"""Seldon model class for Iris inference."""
import errno
import logging
import pickle  # noqa - S403
import sys

from config import app_cfg

import pandas as pd


class Iris:
    """Iris inference model class."""

    def __init__(self):
        """Seldon class init."""
        self.loaded = False
        self.model_name = app_cfg.model.name
        self.model_file = app_cfg.model.file

    def load(self):
        """Load machine learning model."""
        logging.info(f"Loading model from {self.model_file}")

        try:
            self.model = pickle.load(open(self.model_file, "rb"))  # noqa - S301
        except IOError:
            logging.exception(f"Unable to load the model file: {self.model_file}")
            sys.exit(errno.ENOENT)

        self.loaded = True

    def predict(self, X, features_names):
        """Prediction method for model class."""
        if not self.loaded:
            self.load()

        df = pd.DataFrame(data=X, columns=features_names)

        pred = self.model.predict(df)
        return pred
