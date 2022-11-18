import os
from pathlib import Path
from unittest import mock

import environ

import pytest


@pytest.fixture
def set_env_vars():
    with mock.patch.dict(
        os.environ,
        {
            "LOG_LEVEL": "DEBUG",
            "MODEL_NAME": "myTestModel",
            "MODEL_FILE": "models/iris-model.pkl",
        },
    ):

        yield


def test_setting_log_level_env_var(set_env_vars):

    from iris_inference_service.config import AppConfig

    app_cfg = environ.to_config(AppConfig)
    assert app_cfg.log.level == "DEBUG"


def test_setting_model_name_env_var(set_env_vars):

    from iris_inference_service.config import AppConfig

    app_cfg = environ.to_config(AppConfig)
    assert app_cfg.model.name == "myTestModel"


def test_setting_model_file_env_var(set_env_vars):

    from iris_inference_service.config import AppConfig

    app_cfg = environ.to_config(AppConfig)
    assert app_cfg.model.file == Path("models/iris-model.pkl")
