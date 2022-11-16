import os
from unittest import mock

import environ

from pathlib import Path

import pytest


# def test_env_file_values():
#     assert app_cfg.log.level == "INFO"
#     assert (
#         app_cfg.kafka.bootstrap
#         == "kafka-cluster-kafka-bootstrap.kafka.svc.cluster.local:9092"
#     )
#     assert app_cfg.kafka.topic_out == "risk-assessment-result"
#     assert app_cfg.mlservice.host == "tbd-risk-assessment.svc.cluster.local:8000"


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


def test_setting_model_file_env_var(set_env_vars):

    from iris_inference_service.config import AppConfig

    app_cfg = environ.to_config(AppConfig)
    assert app_cfg.model.file == Path("models/iris-model.pkl")


# def test_setting_kafka_bootstrap_env_var(set_env_vars):
#     app_cfg = environ.to_config(AppConfig)
#     assert app_cfg.kafka.bootstrap == "http://kafka-bootstrap-mock.svc:9092"


# def test_setting_kafka_topic_out_env_var(set_env_vars):
#     app_cfg = environ.to_config(AppConfig)
#     assert app_cfg.kafka.topic_out == "mock-kafka-topic"


# def tests_setting_mlservice_host_env_var(set_env_vars):
#     app_cfg = environ.to_config(AppConfig)
#     assert app_cfg.mlservice.host == "http://mock-mlservice.svc:8000"
